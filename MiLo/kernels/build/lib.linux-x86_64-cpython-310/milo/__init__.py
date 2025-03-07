# Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
import torch.nn as nn


import milo_cuda


def mul_3bit(A, B1, B2, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    """Marlin FP16xINT3 multiply; can be used within `torch.compile`.
    @A: `torch.half` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int` weight matrix of original shape `(k, n)` in Marlin format; see `Layer.pack()`
    @C: `torch.half` out matrix of shape `(m, n)` in standard row-major layout
    @s: `torch.half` scales of shape `(m / groupsize, n)`
    @workspace: `torch.int` tensor with at least `n / 128 * max_par` entries that are all zero
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    """
    milo_cuda.mul_3bit(A, B1, B2, C, s, workspace, thread_k, thread_n, sms, max_par)


def mul_3bit_with_zeros(A, B1, B2, C, s, z, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    """Marlin FP16xINT3 multiply; can be used within `torch.compile`.
    @A: `torch.half` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int` weight matrix of original shape `(k, n)` in Marlin format; see `Layer.pack()`
    @C: `torch.half` out matrix of shape `(m, n)` in standard row-major layout
    @s: `torch.half` scales of shape `(m / groupsize, n)`
    @workspace: `torch.int` tensor with at least `n / 128 * max_par` entries that are all zero
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    """
    milo_cuda.mul_3bit_with_zeros(A, B1, B2, C, s, z, workspace, thread_k, thread_n, sms, max_par)



def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single

_perm, _scale_perm, _scale_perm_single = _get_perms()

class Layer3bitWithZeros(nn.Module):
    """PyTorch compatible Marlin layer; 3-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=64, sms = 108):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be 64)
        """
        super().__init__()
        if groupsize != 64 :
            raise ValueError('Only groupsize 64 is supported.')
        if (infeatures % 64 != 0) or (outfeatures % 64 != 0) or (infeatures % 256 != 0 and outfeatures % 256 != 0 ):
            raise ValueError('(64|infeatures & 256|outfeatures)  or (256|infeatures & 64|outfeatures)')
        if infeatures % groupsize != 0:
            raise ValueError('`infeatures` must be divisible by `groupsize`.')
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.tile_shape=0
        thread_block_num = self.k * self.n // (sms  * 64 * 256) + 1;
        self.register_buffer('B1', torch.empty((self.k // 16,2 * self.n * 16 // 32 ), dtype=torch.int))
        self.register_buffer('B2', torch.empty((self.k // 16, self.n * 16 // 32 ), dtype=torch.int))
        self.register_buffer('s', torch.empty((self.k // groupsize, self.n), dtype=torch.half))
        self.register_buffer('z', torch.empty((self.k // groupsize, self.n), dtype=torch.half))
        self.register_buffer('workspace', torch.zeros(self.n // 128 * 16, dtype=torch.int), persistent=False)

    def forward(self, A):
        C = torch.empty(A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device)
        mul_3bit_with_zeros(A.view((-1, A.shape[-1])), self.B1, self.B2, C.view((-1, C.shape[-1])), self.s, self.z, self.workspace)
        return C

    def pack(self, linear, scales, zeros):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """ 
        if linear.weight.dtype != torch.half:
            raise ValueError('Only `torch.half` weights are supported.')
        tile = 16
        maxq = 2 ** 3 - 1
        s = scales.t()
        z = zeros.t()
        w = linear.weight.data.t() # (in-features, out-features) (k, n)
        w = w.reshape((-1, self.groupsize, self.n)) # (k, n) -> (k/group_size, self.group_size, n)
        w = w.permute(1, 0, 2) #(self.group_size, k/group_size, n)
        w = w.reshape((self.groupsize, -1)) # (self.group_size, k/group_size * n)
        s = s.reshape((1, -1)) 
        z = z.reshape((1, -1)) 
        w = torch.round((w - z) / s).int()
        w = torch.clamp(w, 0, maxq)
        w = w.reshape((self.groupsize, -1, self.n)) # (group_size, k/group_size, n)
        w = w.permute(1, 0, 2)
        w = w.reshape((self.k, self.n)).contiguous()
        s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        s = s.reshape((-1, self.n)).contiguous()
        z = z.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        z = z.reshape((-1, self.n)).contiguous()
        
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile)) 
        res = w
        res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
        #else:
        
        q1 = np.zeros((res.shape[0], 2 * res.shape[1]//32), dtype=np.uint32)
        q2 = np.zeros((res.shape[0], res.shape[1]//32), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for j in range(4):
            q1[:,0::2] |= res[:,j::32] << 3*j
            q1[:,0::2] |= res[:,4 + j::32] << 16 + 3*j
            q1[:,1::2] |= res[:,8 + j::32] << 3*j
            q1[:,1::2] |= res[:,12 + j::32] << 16 + 3*j
            q2 |= res[:,16 + j::32] << 3*j
            q2 |= res[:,20 + j::32] << 16 + 3*j
        q1[:,0::2] |= res[:,24::32] << 12
        q1[:,0::2] |= res[:,28::32] << 28
        q1[:,0::2] |= (res[:,25::32] & 0x1) << 15
        q1[:,0::2] |= (res[:,29::32] & 0x1) << 31
        q1[:,1::2] |= (res[:,25::32] & 0x6) << 11
        q1[:,1::2] |= (res[:,26::32] & 0x3) << 14
        q1[:,1::2] |= (res[:,29::32] & 0x6) << 27
        q1[:,1::2] |= (res[:,30::32] & 0x3) << 30
        q2 |= (res[:,26::32] & 0x4) << 10
        q2 |= (res[:,30::32] & 0x4) << 26
        q2 |= res[:,27::32]<< 13
        q2 |= res[:,31::32]<< 29

        q1 = torch.from_numpy(q1.astype(np.int32)).to(w.device)
        q2 = torch.from_numpy(q2.astype(np.int32)).to(w.device)
        self.B1[:, :] = q1.to(self.B1.device)
        self.B2[:, :] = q2.to(self.B2.device)
        self.s[:, :] = s.to(self.s.device)
        self.z[:, :] = z.to(self.z.device)


class Layer3bit(nn.Module):
    """PyTorch compatible Marlin layer; 3-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=64, sms = 108):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be 64)
        """
        super().__init__()
        if groupsize != 64 :
            raise ValueError('Only groupsize 64 is supported.')
        if (infeatures % 64 != 0) or (outfeatures % 64 != 0) or (infeatures % 256 != 0 and outfeatures % 256 != 0 ):
            raise ValueError('(64|infeatures & 256|outfeatures)  or (256|infeatures & 64|outfeatures)')
        if infeatures % groupsize != 0:
            raise ValueError('`infeatures` must be divisible by `groupsize`.')
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.tile_shape=0
        thread_block_num = self.k * self.n // (sms  * 64 * 256) + 1;
        self.register_buffer('B1', torch.empty((self.k // 16,2 * self.n * 16 // 32 ), dtype=torch.int))
        self.register_buffer('B2', torch.empty((self.k // 16, self.n * 16 // 32 ), dtype=torch.int))
        self.register_buffer('s', torch.empty((self.k // groupsize, self.n), dtype=torch.half))
        self.register_buffer('workspace', torch.zeros(self.n // 128 * 16, dtype=torch.int), persistent=False)

    def forward(self, A):
        C = torch.empty(A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device)
        mul_3bit(A.view((-1, A.shape[-1])), self.B1, self.B2, C.view((-1, C.shape[-1])), self.s, self.workspace)
        return C

    def pack(self, linear, scales):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """ 
        if linear.weight.dtype != torch.half:
            raise ValueError('Only `torch.half` weights are supported.')
        tile = 16
        maxq = 2 ** 3 - 1
        s = scales.t()
        w = linear.weight.data.t() # (in-features, out-features) (k, n)
        w = w.reshape((-1, self.groupsize, self.n)) # (k, n) -> (k/group_size, self.group_size, n)
        w = w.permute(1, 0, 2) #(self.group_size, k/group_size, n)
        w = w.reshape((self.groupsize, -1)) # (self.group_size, k/group_size * n)
        s = s.reshape((1, -1)) 
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)
        w = w.reshape((self.groupsize, -1, self.n)) # (group_size, k/group_size, n)
        w = w.permute(1, 0, 2)
        w = w.reshape((self.k, self.n)).contiguous()
        s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        s = s.reshape((-1, self.n)).contiguous()
        
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile)) 
        res = w
        res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
        #else:
        
        q1 = np.zeros((res.shape[0], 2 * res.shape[1]//32), dtype=np.uint32)
        q2 = np.zeros((res.shape[0], res.shape[1]//32), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for j in range(4):
            q1[:,0::2] |= res[:,j::32] << 3*j
            q1[:,0::2] |= res[:,4 + j::32] << 16 + 3*j
            q1[:,1::2] |= res[:,8 + j::32] << 3*j
            q1[:,1::2] |= res[:,12 + j::32] << 16 + 3*j
            q2 |= res[:,16 + j::32] << 3*j
            q2 |= res[:,20 + j::32] << 16 + 3*j
        q1[:,0::2] |= res[:,24::32] << 12
        q1[:,0::2] |= res[:,28::32] << 28
        q1[:,0::2] |= (res[:,25::32] & 0x1) << 15
        q1[:,0::2] |= (res[:,29::32] & 0x1) << 31
        q1[:,1::2] |= (res[:,25::32] & 0x6) << 11
        q1[:,1::2] |= (res[:,26::32] & 0x3) << 14
        q1[:,1::2] |= (res[:,29::32] & 0x6) << 27
        q1[:,1::2] |= (res[:,30::32] & 0x3) << 30
        q2 |= (res[:,26::32] & 0x4) << 10
        q2 |= (res[:,30::32] & 0x4) << 26
        q2 |= res[:,27::32]<< 13
        q2 |= res[:,31::32]<< 29

        q1 = torch.from_numpy(q1.astype(np.int32)).to(w.device)
        q2 = torch.from_numpy(q2.astype(np.int32)).to(w.device)
        self.B1[:, :] = q1.to(self.B1.device)
        self.B2[:, :] = q2.to(self.B2.device)
        self.s[:, :] = s.to(self.s.device)


def replace_linear(module, name_filter=lambda n: True, groupsize=-1, name=''):
    """Recursively replace all `torch.nn.Linear` layers by empty Marlin layers.
    @module: top-level module in which to perform the replacement 
    @name_filter: lambda indicating if a layer should be replaced
    @groupsize: marlin groupsize
    @name: root-level name
    """
    if isinstance(module, Layer):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if isinstance(tmp, nn.Linear) and name_filter(name1):
            setattr(
                module, attr, Layer(tmp.in_features, tmp.out_features, groupsize=groupsize)
            )
    for name1, child in module.named_children():
        replace_linear(child, name_filter, groupsize=groupsize, name=name + '.' + name1 if name != '' else name1)

