import torch
from mamba_ssm import Mamba2

batch, length, dim = 2, 64, 256
x = torch.randn(batch, length, dim).to("cuda")

class MambaModel(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(Mamba2(
                d_model=dim,
                d_state=16,
                d_conv=4,
                expand=2,
            ).to("cuda"))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = MambaModel(4)
y = model(x)
assert y.shape == x.shape

print(model.layers)