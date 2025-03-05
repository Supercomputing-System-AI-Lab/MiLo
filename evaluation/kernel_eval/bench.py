import sys

import numpy as np
import torch
import torch.nn as nn
import milo

# Get the location of the marlin module

import time

def benchmark(f, warmup=20, iter=500):
    for i in range(warmup + iter):
        f()
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    time.sleep(1.)
    return res

def get_problem(m, n, k, groupsize=128):
    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.ones((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B, C, B_ref, s

def get_problem_int3(m, n, k, groupsize=64, tile_shape=0):
    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B1 = torch.randint(low=-2**31, high=2**31, size=(k * n // 16,), device=dev)
    B2 = torch.randint(low=-2**31, high=2**31, size=(k * n // 32,), device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    z = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B1, B2, C, B_ref, s, z

def benchmark_dense(A, B, C):
    res = benchmark(lambda: torch.matmul(A, B, out=C))
    return {
        's': res,
        'TFLOP': (2 * A.numel() * C.shape[1])/ 10 ** 12,
        'GB': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / 10 ** 9
    }

def benchmark_quant(A, B1, B2, C, s, z,thread_k, thread_n, sms):
    workspace = torch.zeros((C.shape[1] // 128) * 16, device=torch.device('cuda:0'))
    res = benchmark(lambda: milo.mul_3bit(A, B1, B2, C, s, workspace, thread_k, thread_n, sms))
    return {
        's': res,
        'TFLOP': (2 * A.numel() * C.shape[1] + 2 * A.shape[1] * C.shape[1])/ 10 ** 12,
        'GB': (2 * A.numel() + 4 * B1.numel() + 4 * B2.numel() + 2 * C.numel() + 2 * s.numel() +  2 * z.numel()) / 10 ** 9
    }

# Pass the SM count for known GPUs to avoid the kernel having to query this information (this is very minor)
gpu = torch.cuda.get_device_name(0)
if 'A100' in gpu:
    SMS = 108
elif 'A10' in gpu:
    SMS = 72
elif '3090' in gpu:
    SMS = 82
elif 'A6000' in gpu:
    SMS = 84
else:
    SMS = -1

MODELS = { #(k,n)
#    'ideal': [
#        (4 * 256 * SMS, 256 * SMS)
# ],
 'deepseek' : [(2048, 11008),(11008,2048),(2048, 11008)],
    'mixtral' : [(4096, 14336),(14336, 4096),(4096, 14336)], 
     'arctic' : [(7168,4864),(4864, 7168),(7168,4864)],
   'Llama65B': [
        (8192, 3 * 8192),
        (8192, 8192),
        (8192, 2 * 21760),
        (21760, 8192)
    ],
   'Falcon180B' : [(14848, 14848 * 5 + 1024),(14848 * 5, 14848)],

#     'deepseek1' : [(2048, 11008)],
#     'deepseek2' : [(11008,2048)],
#    'mixtual1' : [(4096, 14336)], 
#    'mixtual2' : [(14336, 4096)],
#    'arctic1' : [(4864, 7168)],
#    'artic2' : [(7168,4864)],
# 'Falcon180B1' : [(14848, 14848 * 5 + 1024)],
# 'falocon180B2' : [(14848 * 5, 14848)]
}

# Set to true in order to run a more complete benchmark sweep; the default is reproduce README experiments


#2.43, 3.44
for groupsize in [64] :
    print()
    dev = torch.device('cuda:0')
    for model, layers in MODELS.items():
        print(model)
        
        batchsizes = [1,16,32]
        for batch in batchsizes: 
            for sms in [108]:
                tot_q = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0,'memory' : 0, 'TFLOP': 0}  
                tot_d = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0,'memory' : 0, 'TFLOP': 0}  

                for layer in layers:
                    #B_ref, B1, B2, s, z = gen_quant3(layer[0], layer[1], groupsize, tile_shape)
                    tile_shape = 0
                    A, B1, B2, C, B_ref, s, z = get_problem_int3(batch, layer[1], layer[0], groupsize, tile_shape)
                    res_d = benchmark_dense(A, B_ref, C)
                    res_q = benchmark_quant(A, B1, B2, C, s,z,128, 128, SMS)
                    tot_q['s'] += res_q['s']
                    tot_q['memory'] += res_q['GB']
                    tot_q['TFLOP'] += res_q['TFLOP']
                    tot_d['s'] += res_d['s']
                    
                tot_q['TFLOP/s']  =  tot_q['TFLOP'] / tot_q['s'] 
                tot_q['GB/s']  =  tot_q['memory'] / tot_q['s']  
                tot_q['speedup'] = tot_d['s'] / tot_q['s']
                print('sms = %04d: ,batch=%04d: s=%.5f, TFLOP/s=%.3f, GB/s=%.3f, speedup=%.2f, memory(MB)=%.2f, parametersnum : %.2f' % (
                    sms,
                    batch,
                    tot_q['s'],
                    tot_q['TFLOP/s'],
                    tot_q['GB/s'],
                    tot_q['speedup'],
                    tot_q['memory'],
                    layers[0][0] * layers[0][1] / 10**9
                ))
        print()
