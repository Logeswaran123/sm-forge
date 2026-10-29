Fix how data types are used. Feed `tl.dot` the problem's fp16, bf16 or fp8 tensors directly and
accumulate in float32; do not convert operands to float32 in the wrapper or before the dot. For
float32 problems, pass `input_precision="ieee"` (or "tf32x3" where the tolerance allows) to every
`tl.dot`, since the default TF32 fails a 1e-4 tolerance. Keep the outputs' dtypes as they are.
