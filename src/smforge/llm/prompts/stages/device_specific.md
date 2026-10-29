Tune for this device's resources: tile shapes, `num_warps`, `num_stages` and register use, so
kernels do not spill and keep enough warps resident per SM. Use the device facts in the prompt
(shared memory per block, registers, SM count) and the per-kernel registers, spills and
occupancy you were given. Keep the algorithm and memory layout.
