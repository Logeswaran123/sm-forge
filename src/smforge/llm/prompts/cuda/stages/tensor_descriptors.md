Improve how tiles reach shared memory. Copy global-to-shared with `cp.async` (`cuda_pipeline.h`'s
`__pipeline_memcpy_async`, or inline PTX) in a 2- to 4-stage pipeline, so the next tiles load
while the current one is computed, within the shared-memory budget per block; wait for a stage
with `cp.async.wait_group` and `__syncthreads()` before reading it. TMA (`cp.async.bulk.tensor`)
needs a tensor map encoded on the host; use it only if a measurement shows it helps.
