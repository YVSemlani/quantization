import torch
import torch.nn as nn

from transformers import MambaConfig, MambaForCausalLM, MambaCache, AutoTokenizer
from transformers.models.mamba.modeling_mamba import MambaBlock
from transformers.generation import TextStreamer
from datasets import load_dataset

from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from functools import partial
from tqdm import tqdm

# Quamba helper fns
def preprocess(conversation, tokenizer, conversation_template, max_tokens, device):
    """
    Preprocess the data by tokenizing.
    """
    all_input_ids = []
    all_label_ids = []
    tokenizer.use_default_system_prompt = False
    messages = conversation["messages"]
    tokenized_messages = tokenizer.apply_chat_template(messages, chat_template=conversation_template, max_length=max_tokens, truncation=True)
    input_ids = torch.LongTensor([tokenized_messages]).to(device) # expand dim
    return input_ids

class ActIdentity(nn.Module):
    def __init__(self, tensor_name):
        super().__init__()
        self.tensor_name = tensor_name

    @torch.no_grad()
    def forward(self, x):
        return x

    def __repr__(self):
        return f"ActIdentity({self.tensor_name})"

def _get_quant_range(n_bits, sym):
    if sym:
        q_min, q_max = -2**(n_bits-1), 2**(n_bits-1)-1
    else:
        q_min, q_max = -2**(n_bits-1), 2**(n_bits-1)
    return q_min, q_max

def _get_uniform_quantization_params(w_max, n_bits, clip_ratio):
    _, q_max = _get_quant_range(n_bits=n_bits, sym=True)
    if clip_ratio < 1.0:
        w_max = w_max * clip_ratio
    scales = w_max / q_max
    return scales

def _get_minmax_quantization_params(w_max, w_min, n_bits, clip_ratio, sym):
    q_min, q_max = _get_quant_range(n_bits=n_bits, sym=sym)
    if sym:
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
        scales = w_max / q_max
        base = torch.zeros_like(scales)
    else:
        assert w_min is not None, "w_min should not be None for asymmetric quantization."
        if clip_ratio < 1.0:
            w_max = w_max * clip_ratio
            w_min = w_min * clip_ratio
        scales = (w_max-w_min).clamp(min=1e-5) / q_max
        base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
        
    return scales.to(torch.float32).clamp(min=1e-6), base.to(torch.float32)

class PerTensorMinmaxObserver:
    def __init__(self, n_bits, clip_ratio, sym):
        self.n_bits = n_bits
        self.clip_ratio = clip_ratio
        self.w_max = None
        self.w_min = None
        self.sym = sym
        self.has_statistic = False

    def update(self, w):
        self.has_statistic = True
        #assert w.dim() == 2, "Observer only support 2d tensor, please handle the shape outside."
        if self.sym:
            comming_max = w.abs().amax().clamp(min=1e-5)
        else:
            comming_max = w.amax()
            comming_min = w.amin()

        if self.w_max is None:
            self.w_max = comming_max
        else:
            self.w_max = torch.max(comming_max, self.w_max)
        
        if not self.sym:
            if self.w_min is None:
                self.w_min = comming_min
            else:
                self.w_min = torch.min(comming_min, self.w_min)
        
        
    def get_quantization_parameters(self):
        assert self.has_statistic, "Please run the invoke the update() once before getting statistic."
        return _get_minmax_quantization_params(
            w_max=self.w_max,
            w_min=self.w_min,
            n_bits=self.n_bits,
            clip_ratio=self.clip_ratio,
            sym=self.sym
        )
        

class PerTensorPercentileObserver:
    def __init__(self, n_bits, clip_ratio, sym,
                 percentile_sigma=0.01, percentile_alpha=0.99999):
        self.n_bits = n_bits
        self.clip_ratio = clip_ratio
        self.sym = sym
        self.w_max = None
        self.w_min = None
        self.has_statistic = False
        self.percentile_sigma = percentile_sigma
        self.percentile_alpha = percentile_alpha

    def update(self, w):
        self.has_statistic = True
        #assert w.dim() == 2, "Observer only support 2d tensor, please handle the shape outside."
        w = w.clone().to(torch.float32) # quantile() input must be float
        if self.sym:
            cur_max = torch.quantile(w.abs().reshape(-1), self.percentile_alpha)
        else:
            cur_max = torch.quantile(w.reshape(-1), self.percentile_alpha)
            cur_min = torch.quantile(w.reshape(-1),
                                        1.0 - self.percentile_alpha)

        if self.w_max is None:
            self.w_max = cur_max
        else:
            self.w_max = self.w_max + self.percentile_sigma * (cur_max - self.w_max)

        if not self.sym:
            if self.w_min is None:
                self.w_min = cur_min
            else:
                self.w_min = self.w_min + self.percentile_sigma * (cur_min - self.w_min)

    def get_quantization_parameters(self):
        assert self.has_statistic, "Please run the invoke the update() once before getting statistic."
        
        return _get_minmax_quantization_params(
            w_max=self.w_max,
            w_min=self.w_min,
            sym=self.sym,
            n_bits=self.n_bits,
            clip_ratio=self.clip_ratio
        )

# load base model

model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", device_map="auto", torch_dtype=torch.bfloat16)

# build tokenizer
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

print(type(model.backbone.layers[0]))
#print(Block)

# STEP 1: calibrate the model and get activation scales

def calibrate_model(model, tokenizer, num_samples=512, seq_len=512, calibration_dataset=None, preprocess_fn=None):
    layers = model.backbone.layers
    is_mamba_block = lambda block: isinstance(block, MambaBlock)
    get_mamba = lambda block: block.mixer
    is_calib_ops = lambda op: isinstance(op, torch.nn.Linear)
    is_x = lambda op: op == "x_proj"
    is_ssm_state = lambda op: op == "dt_proj"
    percentile_alpha=0.9995 # for smaller model like 130m, use 0.99999
    
    # register min/max observers, num_layer + lm_head
    observers = [{} for _ in range(len(layers) + 1)]

    def stat_hook(m, inputs, outputs, op, block_idx):
        # register the new information to observer
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        observers[block_idx][op + ":input"].update(inputs.clone().detach())

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        observers[block_idx][op + ":output"].update(outputs.clone().detach())

    hooks = []
    for idx, layer in enumerate(layers):
        # skip non-mamba blocks
        if not is_mamba_block(layer):
            print(f"Skipping non-mamba block {idx}")
            continue
        
        # get the mamba mixer
        mixer = get_mamba(layer)
        print(f"Layer {idx}: Found mixer {type(mixer)}")
        for name, m in mixer.named_modules():
            print(f"  Module: {name} -> {type(m)}")
            if is_calib_ops(m):
                print(f"    Registering hook for {name}")
                a_bits = 8
                a_clip_ratio = 1.0
                op = name.split(".")[0]
                if is_x(op) or is_ssm_state(op):
                    observers[idx][op + ":input"] = PerTensorPercentileObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        percentile_alpha=percentile_alpha
                    )
                else:
                    observers[idx][op + ":input"] = PerTensorMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True
                    )
                observers[idx][op + ":output"] = PerTensorMinmaxObserver(
                    n_bits=a_bits,
                    clip_ratio=a_clip_ratio,
                    sym=True
                )
                hooks.append(
                    m.register_forward_hook(partial(stat_hook, op=op, block_idx=idx))
                )
    a_bits = 8
    a_clip_ratio = 1.0

    # add observer hook for lm_head
    observers[-1]["lm_head:input"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    observers[-1]["lm_head:output"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    hooks.append(
        model.lm_head.register_forward_hook(partial(stat_hook, op="lm_head", block_idx=-1))
    )
    
    device = next(model.parameters()).device
    if calibration_dataset is None:
        print("Calibrate with monology/pile-uncopyrighted")
        # Try loading without specifying the compressed file
        try:
            calibration_dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
            # Convert streaming dataset to regular dataset for indexing
            calibration_dataset = calibration_dataset.take(num_samples * 2)  # Take more samples than needed
            calibration_dataset = list(calibration_dataset)
        except Exception as e:
            print(f"Failed to load pile dataset: {e}")
            print("Using a simple text dataset instead")
            # Fallback to a simple dataset
            calibration_dataset = [{"text": "This is a sample text for calibration. " * 50} for _ in range(num_samples)]

        def preprocess(data, tokenizer, max_tokens, device):
            if isinstance(data, dict) and "text" in data:
                text = data["text"]
            else:
                text = str(data)
            input_ids = tokenizer(text, return_tensors="pt",
                    max_length=max_tokens, truncation=True).input_ids.to(device)
            return input_ids
        preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_tokens=seq_len, device=device)

    print("Run calibration")
    for i in tqdm(range(num_samples)):
        input_ids = preprocess_fn(calibration_dataset[i])
        # prepare cache for getting ssm_state scales
        prompt_len = input_ids.size(1)
        # do not set num_last_tokens because we want all activations to lm_head
        model(input_ids)
    
    for h in hooks:
        h.remove()
    
    # collect in/output scaling factors for layers, num_layer + lm_head
    act_scales = [{} for _ in range(len(layers) + 1)]
    for i in range(len(layers) + 1):
        for name, observer in observers[i].items():
            if observer.has_statistic:
                scale, base = observer.get_quantization_parameters()
                # FIXME (HY): hardcode to not use base now
                act_scales[i][name] = scale.to(torch.float32)
            else:
                pass
                #print(f"Warning: Observer {name} in layer {i} has no statistics, skipping...")
    del observers
    return act_scales

act_scales = calibrate_model(model, tokenizer, num_samples=512, seq_len=512, calibration_dataset=None, preprocess_fn=None)
print(act_scales)

# STEP 2: Write quantization functions

def quantize_tensor(tensor, scale, bits=8):
    q_max = 2**(bits-1) - 1
    q_min = -2**(bits-1)
    quantized = torch.round(tensor / scale).clamp(q_min, q_max).to(torch.int8)
    return quantized

def dequantize_tensor(tensor, scale):
    if isinstance(scale, torch.Tensor):
        scale = scale.to(tensor.device)
    return tensor.float() * scale

# STEP 3: Implement custom linear layer with integer operations

class QuantizedLinear(torch.nn.Module):
    def __init__(self, original_linear, w_scale, w_bits=8):
        super().__init__()

        original_linear.weight.data = quantize_tensor(
            original_linear.weight, w_scale, w_bits
        )
        
        # Store quantized weights as integers
        self.weight_quantized, self.weight_scale = quantize_tensor(
            original_linear.weight, w_scale, w_bits
        )
        self.bias = original_linear.bias
        
    def forward(self, x):
        # Dequantize weights for computation
        weight_float = dequantize_tensor(self.weight_quantized, self.weight_scale)

        # TO-DO: Implement int8 multiplication and addition

        return torch.nn.functional.linear(x, weight_float, self.bias)

# STEP 4.1: Implement static quantization

def apply_static_quantization(model, act_scales, w_bits=8, a_bits=8):
    """
    Apply static quantization using pre-computed activation scales.
    """
    layers = model.backbone.layers
    for layer_idx, layer in enumerate(layers):
        if hasattr(layer, 'mixer'):
            for module in layer.mixer.modules():
                if isinstance(module, torch.nn.Linear):
                    if module.in_proj == "x_proj":
                        layers[layer_idx].mixer.in_proj = QuantizedLinear(module.weight, act_scales[layer_idx]["x_proj:input"], w_bits=w_bits)
                    elif module.in_proj == "dt_proj":
                        layers[layer_idx].mixer.dt_proj = QuantizedLinear(module.weight, act_scales[layer_idx]["dt_proj:input"], w_bits=w_bits)
                    else:
                        raise ValueError(f"Unknown in_proj type: {module.in_proj}")
                    
                    layers[layer_idx].mixer.out_proj = QuantizedLinear(module.weight, act_scales[layer_idx]["out_proj:output"], w_bits=w_bits)
    return model