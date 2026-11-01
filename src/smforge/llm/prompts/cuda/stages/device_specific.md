Tune for this device's resources: threads per block, registers and shared memory per block, so
kernels do not spill and keep enough warps resident per SM. Use the device facts in the prompt
(shared memory per block, registers, SM count) and the per-kernel registers, spills and occupancy
you were given; `__launch_bounds__(threads, min_blocks)` caps registers. Use the instructions the
device has: tensor cores through `mma.sync`, WMMA or CUTLASS's SM80 kernels for fp16 and bf16
matrix products, `cp.async` for global-to-shared copies, `ldmatrix` for shared-to-register
fragments. Keep the algorithm and memory layout.
