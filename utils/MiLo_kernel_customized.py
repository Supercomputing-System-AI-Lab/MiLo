import argparse
import numpy as np
import torch
import torch.nn as nn
import milo
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

def get_problem_int3(m, n, k, groupsize=64):
    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B1 = torch.randint(low=-2**31, high=2**31, size=(k * n // 16,), device=dev)
    B2 = torch.randint(low=-2**31, high=2**31, size=(k * n // 32,), device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B1, B2, C, B_ref, s

def benchmark_dense(A, B, C):
    res = benchmark(lambda: torch.matmul(A, B, out=C))
    return {
        's': res,
        'TFLOP/s': (2 * A.numel() * C.shape[1])/res/ 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel())/res / 10 ** 9
    }

def benchmark_quant(A, B1, B2, C, s, thread_k, thread_n, sms):
    workspace = torch.zeros((C.shape[1] // 128) * 16, device=torch.device('cuda:0'))
    res = benchmark(lambda: milo.mul_3bit(A, B1, B2, C, s, workspace, thread_k, thread_n, sms))
    return {
        's': res,
        'TFLOP/s': (2 * A.numel() * C.shape[1])/res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B1.numel() + 4 * B2.numel() + 2 * C.numel() + 2 * s.numel())/res/ 10 ** 9
    }

def main():
    parser = argparse.ArgumentParser(description="Use MiLo kernel to perform FP16xINT3 GeMM with customized settings")
    parser.add_argument('--batchsize', type=int, required=True, help="Batch Size")
    parser.add_argument('--weight_output_dimension', type=int, required=True, help="weight_output_dimension")
    parser.add_argument('--weight_input_dimension', type=int, default=None, help="weight_intput_dimension")
    parser.add_argument('--tile_shape', type=str, choices=["64,256", "128,128"], default="128,128", help="Set tile shape. Options: '64,256', '128,128', '256,64'")

    args = parser.parse_args()
    print(f"Selected tile shape: {args.tile_shape}")

    # Extract m, n, k from model-dimension argument
    m = args.batchsize
    n = args.weight_output_dimension
    k = args.weight_input_dimension
    thread_k, thread_n = map(int, args.tile_shape.split(','))

    # Identify GPU model and set SM count
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
        SMS = -1  # Unknown GPU

    print(f"Detected GPU: {gpu}, SMS count: {SMS}")

    # Create test matrices
    A, B1, B2, C, B_ref, s = get_problem_int3(m, n, k, groupsize=64)

    # Run benchmarks
    print("\n Pytorch matmul results...")
    dense_results = benchmark_dense(A, B_ref, C)
    print(f"Dense: {dense_results['s']:.6f} sec, {dense_results['TFLOP/s']:.2f} TFLOP/s, {dense_results['GB/s']:.2f} GB/s")

    print("\n MiLo kernel matmul performance results...")
    quant_results = benchmark_quant(A, B1, B2, C, s, thread_k, thread_n, sms=SMS)
    print(f"Quantized: {quant_results['s']:.6f} sec, {quant_results['TFLOP/s']:.2f} TFLOP/s, {quant_results['GB/s']:.2f} GB/s")

if __name__ == "__main__":
    main()
