Fix how work is spread over the SMs. When there are fewer blocks than SMs, split the reduction
dimension across blocks (split-K, or split-KV for decode attention) and combine the partials in
a second kernel or with `atomicAdd` on a buffer zeroed on the stream (`cudaMemsetAsync`) or by
the kernel. When the last wave is mostly idle, change the tile size or use a persistent grid of
SMs times resident blocks, each looping over tiles.
