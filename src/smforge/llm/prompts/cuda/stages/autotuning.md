Change only how the existing kernels are launched: threads per block, tile sizes, elements per
thread, pipeline stages, split factors, and the grids that follow from them. Make them template
or `constexpr` parameters, instantiate 2 to 6 candidates that fit the shared-memory and register
budgets, and have the host launcher pick one per call by a rule on the sizes (no timing at run
time, no cached choice). For a CUTLASS kernel the parameters are its threadblock, warp and
instruction shapes, stage count and swizzle. Keep each kernel's algorithm, arithmetic and memory
layout, and keep `__launch_bounds__` equal to the threads per block.
