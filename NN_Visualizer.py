from torchviz import make_dot
import torch
import os
from models.MLP import MLP

input_size   = 784
hidden_sizes = [512, 256, 128]
num_classes  = 10

model = MLP(input_size=input_size, hidden_sizes=hidden_sizes, num_classes=num_classes)
model.eval()
x = torch.randn(1, input_size)

os.makedirs("plots", exist_ok=True)

arch_str  = "_".join(str(h) for h in hidden_sizes)
file_name = f"plots/mlp_{input_size}_{arch_str}_{num_classes}"

dot = make_dot(model(x), params=dict(model.named_parameters()))
dot.render(file_name, format="png", cleanup=True)

print(f"Saved: {file_name}.png")
