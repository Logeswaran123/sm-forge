Fix how data types are used. Load fp16 and bf16 data as such, two or eight elements per load
(`__half2`, `__nv_bfloat162`, 16-byte vectors), feed tensor cores the fp16 or bf16 operands
directly and accumulate in float; do not convert whole tensors to float32 in the binding. For
float32 inputs compute in float32, not TF32 or half precision, so a 1e-4 tolerance holds. Keep
the outputs' dtypes as they are.
