import unittest

import numpy as np
import torch
import torch.nn as nn
import milo

torch.cuda.empty_cache()  # Clears the GPU cache

seed = 90 #16 llama
print("seed : ",seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device('cuda:0')

def gen_quant3(m, n, groupsize=64):
    maxq = 2 ** 3 - 1
    w = torch.ones((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    z = torch.randn(s.shape, dtype=torch.half,device=DEV)
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    #ref = w.half() * s + z
    #ref = w
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)

    s = s.reshape((-1, n)).contiguous()
    z = z.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    layer = milo.Layer3bit(m, n, groupsize=groupsize)    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B1 = torch.empty((m // 16, n * 16 * 2 // 32), dtype=torch.int, device=DEV)
    layer.B2 = torch.empty((m // 16, n * 16 // 32), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    q1 = layer.B1
    q2 = layer.B2
    s = layer.s
    return ref, q1, q2, s 

def gen_quant3_with_zero(m, n, groupsize=64):
    maxq = 2 ** 3 - 1
    w = torch.randn((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    z = torch.randn(s.shape, dtype=torch.half,device=DEV)
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    #ref = w.half() * s + z
    #ref = w
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)

    s = s.reshape((-1, n)).contiguous()
    z = z.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    layer = milo.Layer3bit(m, n, groupsize=groupsize)
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B1 = torch.empty((m // 16, n * 16 * 2 // 32), dtype=torch.int, device=DEV)
    layer.B2 = torch.empty((m // 16, n * 16 // 32), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.z = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    #layer.pack(linear, s.t(), z.t())
    layer.pack(linear, s.t())
    q1 = layer.B1
    q2 = layer.B2
    s = layer.s
    z = layer.z
    return ref, q1, q2, s, z


class Test(unittest.TestCase):

    def run_problem_with_zero(self, m, n, k, thread_k, thread_n, groupsize=64):  # 16, 512, 768, 64, 256
        print('% 5d % 6d % 6d % 4d % 4d % 4d' % (m, n, k, thread_k, thread_n, groupsize))
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        B_ref, B1, B2, s, z = gen_quant3_with_zero(k, n, groupsize=groupsize)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        C_ref = torch.matmul(A, B_ref)
        workspace = torch.zeros(n // 128 * 16, device=DEV)
        milo.mul_3bit(A, B1, B2, C, s, workspace, thread_k, thread_n)
        torch.cuda.synchronize()
        self.assertLess(torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)), 0.005)

    def run_problem(self, m, n, k, thread_k, thread_n, groupsize=64):  # 16, 512, 768, 64, 256
        print('% 5d % 6d % 6d % 4d % 4d % 4d' % (m, n, k, thread_k, thread_n, groupsize))
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        B_ref, B1, B2, s = gen_quant3(k, n, groupsize=groupsize)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        C_ref = torch.matmul(A, B_ref)
        workspace = torch.zeros(n // 128 * 16, device=DEV)
        milo.mul_3bit(A, B1, B2, C, s, workspace, thread_k, thread_n)
        torch.cuda.synchronize()
        self.assertLess(torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)), 0.005)

    def test_moe(self):
        for m in range(1,17):
            self.run_problem(m,4096, 14336, thread_k, thread_n,64) 
  
    def test_tiles(self):
        print("test_tiles")
        for m in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 118, 128]:
            for thread_k, thread_n in [(64, 256),(256, 64),(128,128)]:
                self.run_problem(m, 2 * 256, 1024, thread_k, thread_n)
    
    def test_k_stages_divisibility(self):    
        print("test_k_stages_divisibility")
        for k in [3*256 + 256 * 4 * 2 + 256 * i for i in range(1, 6, 2)]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_very_few_stages(self):
        print("test_very_few_stages")
        for k in [ 128, 256, 384]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_llama_shapes(self):
        return
        print("test_llama_shapes")
        MODELS = {
            ' 7B': [
                (4096, 3 * 4096),
                (4096, 4096),
                (4096, 2 * 10752),
                (10752, 4096)
            ],
            '13B': [
                (5120, 3 * 5120),
                (5120, 5120),
                (5120, 2 * 13568),
                (13568, 5120)
            ],
            '33B': [
                (6656, 3 * 6656),
                (6656, 6656),
                (6656, 2 * 17664),
                (17664, 6656)
            ],
            '70B': [
                (8192, 3 * 8192),
                (8192, 8192),
                (8192, 2 * 21760),
                (21760, 8192)
            ]
        }

        for _, layers in MODELS.items():
            for layer in layers:
                for thread_k, thread_n in [(64, 256),(128, 128),(256, 64)]:
                    for batch in [1, 16]:
                        self.run_problem(batch, layer[1], layer[0],thread_k, thread_n,64)

    def test_errors(self):       
        print("test_errors")
        m, n, k = 16, 256, 128
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        B_ref, B1, B2, s = gen_quant3(k, n, groupsize=64)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        workspace = torch.zeros(n // 128 , device=DEV)
        err = False
        try:
            #marlin.mul_3bit(A, B1, B2, C, s, workspace, 256, 256, -1)
            marlin.mul_3bit_faster(A, B1, B2, C, s, workspace,64)
        except:
            err = True 
        self.assertTrue(err)
        s = torch.zeros((2, n), dtype=torch.half, device=DEV)
        err = False
        try:
            #marlin.mul_3bit(A, B1, B2, C, s, workspace, 256, 256, -1)
            marlin.mul_3bit_faster(A, B1, B2, C, s, workspace, 64)
        except:
            err = True 
        self.assertTrue(err)
    

if __name__ == '__main__':
    unittest.main()
