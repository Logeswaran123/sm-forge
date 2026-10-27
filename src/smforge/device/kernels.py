"""Triton kernels for the feature probes and the read-bandwidth measurement."""

import triton
import triton.language as tl


@triton.jit
def dot_kernel(
    a, b, c, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(a + offs_m[:, None] * K + (k + offs_k)[None, :])
        y = tl.load(b + (k + offs_k)[:, None] * N + offs_n[None, :])
        acc = tl.dot(x, y, acc)
    tl.store(c + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def warp_specialize_kernel(
    a, b, c, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K, warp_specialize=True):
        x = tl.load(a + offs_m[:, None] * K + (k + offs_k)[None, :])
        y = tl.load(b + (k + offs_k)[:, None] * N + offs_n[None, :])
        acc = tl.dot(x, y, acc)
    tl.store(c + offs_m[:, None] * N + offs_n[None, :], acc)


@triton.jit
def tma_host_kernel(
    a_desc, b_desc, c_desc, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        acc = tl.dot(a_desc.load([pid_m * BLOCK_M, k]), b_desc.load([k, pid_n * BLOCK_N]), acc)
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc)


@triton.jit
def tma_device_kernel(
    a, b, c, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    a_desc = tl.make_tensor_descriptor(a, [M, K], [K, 1], [BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b, [K, N], [N, 1], [BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(c, [M, N], [N, 1], [BLOCK_M, BLOCK_N])
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        acc = tl.dot(a_desc.load([pid_m * BLOCK_M, k]), b_desc.load([k, pid_n * BLOCK_N]), acc)
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc)


@triton.jit
def dot_scaled_kernel(
    a, b, a_scale, b_scale, c, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_s = tl.arange(0, BLOCK_K // 32)
    x = tl.load(a + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    y = tl.load(b + offs_k[:, None] * BLOCK_N + offs_n[None, :])
    x_scale = tl.load(a_scale + offs_m[:, None] * (BLOCK_K // 32) + offs_s[None, :])
    y_scale = tl.load(b_scale + offs_n[:, None] * (BLOCK_K // 32) + offs_s[None, :])
    acc = tl.dot_scaled(x, x_scale, "e4m3", y, y_scale, "e4m3")
    tl.store(c + offs_m[:, None] * BLOCK_N + offs_n[None, :], acc)


@triton.jit
def read_kernel(x, out, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offs, mask=offs < n, other=0.0)
    tl.store(out + tl.program_id(0), tl.sum(values, axis=0))
