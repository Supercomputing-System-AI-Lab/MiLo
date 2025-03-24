import torch
api_ver = torch.load("/home/exouser/MiLo/MiLo_api/layer2_downproj.pt")
good_ver = torch.load("/home/exouser/MiLo/MiLo_v0/goodver.pt")
a = api_ver["W_unquant"]
a2 = good_ver["W_unquant"]
print(a)
print(a2)
print(a-a2)
# (U_scale,U_packed),(V_scale,V_packed)
