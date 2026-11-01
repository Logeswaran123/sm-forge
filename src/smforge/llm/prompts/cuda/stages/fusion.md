Merge work into fewer kernels. Apply elementwise ops, bias, scales and reductions in the epilogue
of the kernel that produces their input instead of storing it and reading it back; compute a
row's statistics and its normalized output in one block; put batch, head or expert indices on a
grid axis instead of launching per item; give a CUTLASS GEMM an epilogue functor
(`LinearCombinationRelu` and the like) instead of a second kernel. Remove host synchronization
from `forward`. Do not fuse so much that registers spill.
