import torch

xs = torch.randn(30)
print(f"{xs}\n"
      f"Standard Deviation: {torch.std(xs)}\n"
      f"Mean: {torch.mean(xs)}")
