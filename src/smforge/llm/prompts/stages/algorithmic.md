Change what is computed, not how it is scheduled: reorder or simplify the math so it does less
work while giving the same result. Typical changes are reducing before a linear op (a sum or
pool after a GEMM or convolution), dropping ops that are no-ops at these shapes or in eval mode,
skipping blocks a causal mask zeroes, and computing a convolution as an implicit GEMM on tensor
cores instead of with scalar multiply-adds. Recompute anything derived from the weights on every
call. The result must match the reference within the stated tolerance for any input.
