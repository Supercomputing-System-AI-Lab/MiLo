import sys

import numpy as np
import torch
import torch.nn as nn
import milo

# Get the location of the marlin module

import time

def benchmark(f, warmup=40, iter=800):
    for i in range(warmup + iter):
        f()
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    time.sleep(1.)
    return res

def get_problem(m, n, k, groupsize=64):
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
        'ms': 1000 * res,
        'TFLOP': (2 * A.numel() * C.shape[1])/ 10 ** 12,
        'GB': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / 10 ** 9
    }

def benchmark_quant(A, B1, B2, C, s, z,thread_k, thread_n, sms, test = 'asymmetirc'):
    workspace = torch.zeros((C.shape[1] // 128) * 16, device=torch.device('cuda:0'))
    if test ==  'asymmetirc':
        res = benchmark(lambda: milo.mul_3bit_with_zeros(A, B1, B2, C, s,z, workspace, thread_k, thread_n, sms))
        return {
            'ms': 1000 * res,
            'TFLOP': (2 * A.numel() * C.shape[1] + 2 * A.shape[1] * C.shape[1])/ 10 ** 12,
            'GB': (2 * A.numel() + 4 * B1.numel() + 4 * B2.numel() + 2 * C.numel() + 2 * s.numel() +  2 * z.numel()) / 10 ** 9
        }
    if test =='dequant_ablation':
        res = benchmark(lambda: milo.mul_dequant_ablation(A, B1, B2, C, s,z, workspace, thread_k, thread_n, sms))
        return {
            'ms': 1000 * res,
            'TFLOP': (2 * A.numel() * C.shape[1] + A.shape[1] * C.shape[1])/ 10 ** 12,
            'GB': (2 * A.numel() + 4 * B1.numel() + 4 * B2.numel() + 2 * C.numel() + 2 * s.numel() +  2 * z.numel()) / 10 ** 9
        }
    if test =='pipeline_ablation':
        res = benchmark(lambda: milo.mul_pipeline_ablation(A, B1, B2, C, s,z, workspace, thread_k, thread_n, sms))
        return {
            'ms': 1000 * res,
            'TFLOP': (2 * A.numel() * C.shape[1] + A.shape[1] * C.shape[1])/ 10 ** 12,
            'GB': (2 * A.numel() + 4 * B1.numel() + 4 * B2.numel() + 2 * C.numel() + 2 * s.numel() +  2 * z.numel()) / 10 ** 9
        }        
    else:
        res = benchmark(lambda: milo.mul_3bit(A, B1, B2, C, s, workspace, thread_k, thread_n, sms))
        return {
            'ms': 1000 * res,
            'TFLOP': (2 * A.numel() * C.shape[1] + A.shape[1] * C.shape[1])/ 10 ** 12,
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
'arctic' : [(7168,4864),(4864, 7168),(7168,4864)],
'mixtral' : [(4096, 14336),(14336, 4096),(4096, 14336)], 
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

for groupsize in [64] :
    print()
    dev = torch.device('cuda:0')
    for model, layers in MODELS.items():
        print(model)
        batchsizes = [16]
        for batch in batchsizes: 
            sms = 108
            #for thread_k, thread_n in [(128, 128),(64, 256),(256, 64)]:
            tot_q = {'ms': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0,'memory' : 0, 'TFLOP': 0}  
            tot_d = {'ms': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0,'memory' : 0, 'TFLOP': 0}  
            for layer in layers:
                if model == 'Falcon180B':
                    thread_k, thread_n = 64, 256
                elif layer[1] * 2 < layer[0] and model not in ['arctic', 'ideal'] and batch <= 16:
                    thread_k, thread_n = 256, 64
                else:
                    thread_k, thread_n = 128, 128
                A, B1, B2, C, B_ref, s, z = get_problem(batch, layer[1], layer[0], groupsize)
                res_d = benchmark_dense(A, B_ref, C)
                res_q = benchmark_quant(A, B1, B2, C, s, z, thread_k, thread_n, SMS)
                tot_q['ms'] += res_q['ms']
                tot_q['memory'] += res_q['GB']
                tot_q['TFLOP'] += res_q['TFLOP']
                tot_d['ms'] += res_d['ms']
                
            tot_q['TFLOP/s']  =  1000 * tot_q['TFLOP'] / tot_q['ms'] 
            tot_q['GB/s']  =  1000 * tot_q['memory'] / tot_q['ms']  
            tot_q['speedup'] = tot_d['ms'] / tot_q['ms']
            print('threadk=%04d,thread_n=%04d: ,batch=%04d: ms=%.5f, TFLOP/s=%.3f, GB/s=%.3f, speedup=%.2f' % (
                thread_k,
                thread_n,
                batch,
                tot_q['ms'],
                tot_q['TFLOP/s'],
                tot_q['GB/s'],
                tot_q['speedup'],
            ))
        print()
