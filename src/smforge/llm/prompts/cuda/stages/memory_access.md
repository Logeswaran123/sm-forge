Improve how kernels touch memory. Map consecutive threads to consecutive addresses so each warp's
loads coalesce; load 16 bytes per thread (`float4`, `uint4`) where the pointer and size allow,
with a scalar tail; read transposed operands through their strides instead of copying them; stage
reused tiles in shared memory and pad their rows (`[TILE][TILE + 1]`) or swizzle them so threads
do not hit the same bank; order the grid so blocks running together share the larger operand;
mark read-only inputs `const __restrict__`.
