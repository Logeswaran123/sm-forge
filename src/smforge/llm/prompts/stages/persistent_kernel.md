Fix how work is spread over the SMs. When there are fewer tiles than SMs, split the reduction
dimension across programs (split-K, or split-KV for decode attention) and combine the partials
in a second kernel or with atomics on a zeroed buffer. When the last wave is mostly idle, change
the tile size or use a persistent grid of one program per SM looping over tiles.
