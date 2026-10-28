Change only how the existing kernels are launched: tile sizes (BLOCK_*), num_warps, num_stages,
group sizes, split factors, and the grids that follow from them. Prefer wrapping each kernel in
`@triton.autotune` over 4 to 12 configs that fit the shared-memory and register budgets, with
`key` naming the shape arguments that change the best config, and make the grid a function of
the meta-parameters. Keep each kernel's algorithm, arithmetic and memory layout. A kernel that
accumulates into an argument needs `reset_to_zero` (or `restore_value`) on that argument.
