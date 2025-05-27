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

# calibrate the model and get activation scales

def calibrate_model(model, tokenizer, num_samples=512, seq_len=512, calibration_dataset=None, preprocess_fn=None):
    layers = model.backbone.layers
    is_mamba_block = lambda block: isinstance(block, MambaBlock)
    get_mamba = lambda block: block.mixer
    is_calib_ops = lambda op: isinstance(op, torch.nn.Linear)
    is_x = lambda op: op == "x_proj"
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
        for name, m in mixer.named_modules():
            if is_calib_ops(m):
                a_bits = 8
                a_clip_ratio = 1.0
                op = name.split(".")[0]
                if is_x(op):
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
        calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")

        def preprocess(data, tokenizer, max_tokens, device):
            input_ids = tokenizer(data["text"], return_tensors="pt",
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
            scale, base = observer.get_quantization_parameters()
            # FIXME (HY): hardcode to not use base now
            act_scales[i][name] = scale.to(torch.float32)
    del observers
    return act_scales

calibrate_model(model, tokenizer, num_samples=512, seq_len=512, calibration_dataset=None, preprocess_fn=None)

"""
@torch.no_grad()
def run_quamba_calibration(
        model, model_type, tokenizer, num_samples=512, seq_len=512,
        calibration_dataset=None, preprocess_fn=None
    ):

    if model_type == "mamba":
        layers = model.backbone.layers
        is_traget_block = lambda block: isinstance(block, Block)
        get_mamba = lambda block: block.mixer
        is_calib_ops = lambda op: isinstance(op, torch.nn.Linear)
        is_x = lambda op: op == "x_proj"
        is_ssm_state = lambda op: op == "ssm_state_act"
        percentile_alpha=0.9995 # for smaller model like 130m, use 0.99999
    elif model_type == "mamba2":
        layers = model.backbone.layers
        is_traget_block = lambda block: isinstance(block, Block)
        get_mamba = lambda block: block.mixer
        is_calib_ops = lambda op: isinstance(op, torch.nn.Linear)
        is_x = lambda op: op == "x_conv_out"
        is_ssm_state = lambda op: op == "ssm_state_act"
        percentile_alpha=0.9995  # for smaller model like 130m, use 0.99999
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")

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
    for i in range(len(layers)):
        if not is_traget_block(layers[i]):
            continue
        
        mixer = get_mamba(layers[i])
        for name, m in mixer.named_modules():
            if is_calib_ops(m):
                # FIXME(HY): hardcode everything for now
                a_bits = 8
                a_clip_ratio = 1.0
                op = name.split(".")[0]
                if is_x(op) or is_ssm_state(op):
                    observers[i][op + ":input"] = PerTensorPercentileObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        percentile_alpha=percentile_alpha
                    )
                else:
                    observers[i][op + ":input"] = PerTensorMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True
                    )
                observers[i][op + ":output"] = PerTensorMinmaxObserver(
                    n_bits=a_bits,
                    clip_ratio=a_clip_ratio,
                    sym=True
                )
                hooks.append(
                    m.register_forward_hook(partial(stat_hook, op=op, block_idx=i))
                )
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
        logger.info("Calibrate with monology/pile-uncopyrighted")
        calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")

        def preprocess(data, tokenizer, max_tokens, device):
            input_ids = tokenizer(data["text"], return_tensors="pt",
                    max_length=max_tokens, truncation=True).input_ids.to(device)
            return input_ids
        preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_tokens=seq_len, device=device)

    logger.info("Run calibration")
    for i in tqdm(range(num_samples)):
        input_ids = preprocess_fn(calibration_dataset[i])
        # prepare inference cache for getting ssm_state scales
        prompt_len = input_ids.size(1)
        inf_cache = model.allocate_inference_cache(1, prompt_len)
        lengths_per_sample = torch.full((1,), prompt_len, dtype=torch.int32, device=device)
        inference_params = InferenceParams(
            max_seqlen=prompt_len,
            max_batch_size=1,
            seqlen_offset=0,
            key_value_memory_dict=inf_cache,
            lengths_per_sample=lengths_per_sample,
        )
        # do not set num_last_tokens because we want all activations to lm_head
        model(input_ids, inference_params=inference_params)
        # clean up the cache
        del inf_cache
    
    for h in hooks:
        h.remove()
    
    # collect in/output scaling factors for layers, num_layer + lm_head
    act_scales = [{} for _ in range(len(layers) + 1)]
    for i in range(len(layers) + 1):
        for name, observer in observers[i].items():
            scale, base = observer.get_quantization_parameters()
            # FIXME (HY): hardcode to not use base now
            act_scales[i][name] = scale.to(torch.float32)
    del observers
    return act_scales
"""