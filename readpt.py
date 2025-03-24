import torch

state_dict = torch.load("/media/volume/MiLo_v3/MiLo_api_test/compensators.pt", map_location="cuda")  # or "cuda"


first_key = "model.layers.27.mlp.shared_experts.down_proj"
first_value = state_dict[first_key]

print(first_key)
print(first_value)
