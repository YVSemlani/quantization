from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
#input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"]

#out = model.generate(input_ids, max_new_tokens=10)
#print(tokenizer.batch_decode(out))

print(model)

size_check = False
if size_check:

    # Number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    # Model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size: {size_all_mb:.2f} MB")

    # Perform quantization
    quantized_model = MambaForCausalLM.from_pretrained(
        "state-spaces/mamba-130m-hf",
        load_in_4bit=True,
        device_map="auto"  # or "cuda" if you want to force GPU
    )

    # Quantized model size in MB
    param_size = 0
    for param in quantized_model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in quantized_model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Quantized model size: {size_all_mb:.2f} MB")