Improve how kernels touch memory. Make the contiguous dimension the fastest-varying one in each
tile so loads coalesce, read transposed operands through their strides instead of copying them,
size tiles so a warp loads long contiguous runs, order the grid so programs running together
share the larger operand, and avoid shared-memory bank conflicts. Read weights in the layout
they come in, through their strides, instead of copying or transposing them.
