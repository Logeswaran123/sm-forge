| Run | Changes | Problems | Improved | Regressions | Incorrect | Over seed | Over eager | Over compile | Over max-autotune | $ per problem | Minutes per problem |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abl-base | baseline | 12 | 12 | 0 | 0 | 2.18x | 4.27x | 2.42x | - | 1.15 | 5.9 |
| abl-no-kb | no KB | 12 | 12 | 0 | 0 | 3.01x | 5.94x | 3.36x | - | 1.08 | 5.7 |
| abl-no-facts | no facts | 12 | 12 | 0 | 0 | 2.10x | 4.12x | 2.33x | - | 0.80 | 4.0 |
| abl-no-reanalysis | no re-analysis | 12 | 12 | 0 | 0 | 2.13x | 4.18x | 2.37x | - | 1.05 | 5.5 |
| abl-no-rollback | no rollback | 12 | 12 | 0 | 0 | 2.20x | 4.31x | 2.44x | - | 0.98 | 5.7 |
| abl-t1 | T = 1 | 12 | 12 | 0 | 0 | 2.01x | 3.95x | 2.23x | - | 0.59 | 4.5 |
| abl-t3 | T = 3 | 12 | 12 | 0 | 0 | 2.27x | 4.48x | 2.53x | - | 1.35 | 6.3 |
| abl-t5 | T = 5 | 12 | 12 | 0 | 0 | 2.21x | 4.33x | 2.45x | - | 2.72 | 8.5 |
| abl-best3 | best-of-3 | 12 | 12 | 0 | 0 | 2.30x | 4.52x | 2.56x | - | 2.80 | 16.2 |
| abl-opus-low | claude-opus-5-5, effort low | 12 | 12 | 0 | 0 | 1.88x | 3.69x | 2.09x | - | 0.80 | 4.4 |
| abl-opus-high | claude-opus-5-5, effort high | 12 | 12 | 0 | 0 | 2.32x | 4.54x | 2.57x | - | 1.82 | 9.7 |
| abl-sonnet | claude-sonnet-5-5, effort medium | 12 | 12 | 0 | 0 | 1.96x | 3.84x | 2.17x | - | 0.43 | 3.6 |
| abl-haiku | claude-haiku-5-5, effort medium | 12 | 12 | 0 | 0 | 1.84x | 3.65x | 2.06x | - | 0.03 | 4.1 |

### abl-base

| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile | Roofline | Held-out checks | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| inf_paged_attention_decode | 1195.0 us | 303.1 us | 3.94x | 4.24x | 28.71x | 13.75x | 1.00 | pass | algorithmic |
| l1_008_matmul_with_irregular_shapes | 4617.2 us | 2636.8 us | 1.75x | 1.75x | 1.89x | 1.05x | 0.94 | pass | open_ended, tensor_descriptors, autotuning |
| l1_035_groupnorm | 1656.9 us | 1114.0 us | 1.49x | 1.57x | 2.25x | 1.48x | 0.95 | kernelbench skipped: out of memory | algorithmic, autotuning |
| l1_049_max_reduction_over_a_dimension | 3765.7 us | 2474.0 us | 1.52x | 1.48x | 1.16x | 1.23x | 0.98 | pass | persistent_kernel, memory_access |
| l2_013_convtranspose3d_mean_add_softmax_tanh_scaling | 829.3 us | 141.2 us | 5.88x | 6.26x | 28.88x | 24.07x | 7.07 | pass | algorithmic, fusion, autotuning |
| l2_023_conv3d_groupnorm_mean | 1120.3 us | 246.8 us | 4.54x | 4.71x | 10.54x | 7.82x | 0.34 | pass | algorithmic, open_ended, autotuning |
| l2_045_gemm_sigmoid_logsumexp | 4497.4 us | 3518.3 us | 1.28x | 1.30x | 1.14x | 1.12x | 1.01 | pass | open_ended, fusion |
| l2_057_conv2d_relu_hardswish | 225.2 us | 189.6 us | 1.19x | 1.21x | 13.93x | 2.70x | 0.87 | pass | open_ended, memory_access, autotuning |
| l2_067_conv2d_gelu_globalavgpool | 1514.1 us | 368.6 us | 4.11x | 4.12x | 7.62x | 1.99x | 0.44 | pass | algorithmic, fusion, autotuning |
| l2_088_gemm_groupnorm_swish_multiply_swish | 1364.0 us | 1163.2 us | 1.17x | 1.18x | 1.27x | 1.09x | 1.02 | pass | fusion, persistent_kernel |
| l2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1823.7 us | 514.8 us | 3.54x | 3.55x | 5.75x | 1.33x | 0.57 | pass | open_ended, autotuning |
| l2_095_matmul_add_swish_tanh_gelu_hardtanh | 1462.3 us | 1366.0 us | 1.07x | 1.09x | 0.97x | 0.92x | 0.87 | pass | memory_access, persistent_kernel |

### abl-no-kb

| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile | Roofline | Held-out checks | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| inf_paged_attention_decode | 1194.0 us | 303.1 us | 3.94x | 4.44x | 28.71x | 13.75x | 1.00 | pass | algorithmic |
| l1_008_matmul_with_irregular_shapes | 4587.5 us | 2720.7 us | 1.69x | 1.59x | 1.83x | 1.02x | 0.91 | pass | open_ended |
| l1_035_groupnorm | 1646.9 us | 1054.7 us | 1.56x | 1.57x | 2.38x | 1.57x | 1.01 | kernelbench skipped: out of memory | algorithmic, autotuning |
| l1_049_max_reduction_over_a_dimension | 3476.7 us | 2409.5 us | 1.44x | 1.30x | 1.19x | 1.26x | 1.01 | pass | persistent_kernel, memory_access |
| l2_013_convtranspose3d_mean_add_softmax_tanh_scaling | 829.5 us | 139.3 us | 5.95x | 6.35x | 29.26x | 24.39x | 7.16 | pass | algorithmic, dtype_fix, autotuning |
| l2_023_conv3d_groupnorm_mean | 1119.2 us | 6.0 us | 188.04x | 534.80x | 437.19x | 324.45x | 14.25 | pass | algorithmic, dtype_fix |
| l2_045_gemm_sigmoid_logsumexp | 4599.9 us | 3580.0 us | 1.28x | 1.19x | 1.13x | 1.10x | 0.99 | pass | open_ended, fusion |
| l2_057_conv2d_relu_hardswish | 225.2 us | 178.2 us | 1.26x | 1.25x | 14.82x | 2.87x | 0.93 | pass | memory_access |
| l2_067_conv2d_gelu_globalavgpool | 1511.5 us | 342.0 us | 4.42x | 4.38x | 8.21x | 2.14x | 0.48 | pass | algorithmic |
| l2_088_gemm_groupnorm_swish_multiply_swish | 1364.0 us | 1243.1 us | 1.10x | 1.10x | 1.19x | 1.02x | 0.95 | pass | fusion, autotuning |
| l2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1829.9 us | 505.9 us | 3.62x | 3.61x | 5.86x | 1.35x | 0.58 | pass | algorithmic, autotuning |
| l2_095_matmul_add_swish_tanh_gelu_hardtanh | 1462.4 us | 1245.2 us | 1.17x | 1.18x | 1.06x | 1.01x | 0.95 | pass | tensor_descriptors, persistent_kernel |

### abl-no-facts

| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile | Roofline | Held-out checks | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| inf_paged_attention_decode | 1209.4 us | 303.1 us | 3.99x | 4.31x | 28.71x | 13.75x | 1.00 | pass | persistent_kernel, algorithmic |
| l1_008_matmul_with_irregular_shapes | 4929.4 us | 2999.1 us | 1.64x | 1.74x | 1.66x | 0.92x | 0.82 | pass | memory_access |
| l1_035_groupnorm | 1651.8 us | 1149.8 us | 1.44x | 1.42x | 2.18x | 1.44x | 0.92 | kernelbench skipped: out of memory | open_ended |
| l1_049_max_reduction_over_a_dimension | 3512.5 us | 2479.3 us | 1.42x | 1.39x | 1.15x | 1.23x | 0.98 | pass | open_ended, memory_access, autotuning |
| l2_013_convtranspose3d_mean_add_softmax_tanh_scaling | 829.4 us | 139.3 us | 5.95x | 6.38x | 29.26x | 24.39x | 7.16 | pass | algorithmic, memory_access, autotuning |
| l2_023_conv3d_groupnorm_mean | 1119.3 us | 294.7 us | 3.80x | 3.92x | 8.83x | 6.55x | 0.29 | pass | algorithmic |
| l2_045_gemm_sigmoid_logsumexp | 4525.9 us | 3578.2 us | 1.26x | 1.16x | 1.13x | 1.10x | 0.99 | pass | fusion |
| l2_057_conv2d_relu_hardswish | 225.2 us | 217.1 us | 1.04x | 1.03x | 12.17x | 2.36x | 0.76 | pass | autotuning |
| l2_067_conv2d_gelu_globalavgpool | 1510.5 us | 370.5 us | 4.08x | 4.20x | 7.58x | 1.98x | 0.44 | pass | algorithmic, open_ended, autotuning |
| l2_088_gemm_groupnorm_swish_multiply_swish | 1364.1 us | 1250.5 us | 1.09x | 1.14x | 1.19x | 1.01x | 0.95 | pass | fusion, autotuning |
| l2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1825.0 us | 506.3 us | 3.60x | 3.56x | 5.85x | 1.35x | 0.58 | pass | open_ended, autotuning |
| l2_095_matmul_add_swish_tanh_gelu_hardtanh | 1462.5 us | 1243.9 us | 1.18x | 1.17x | 1.06x | 1.01x | 0.95 | pass | autotuning |

### abl-no-reanalysis

| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile | Roofline | Held-out checks | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| inf_paged_attention_decode | 1195.0 us | 302.9 us | 3.94x | 4.30x | 28.73x | 13.76x | 1.00 | pass | algorithmic |
| l1_008_matmul_with_irregular_shapes | 4567.9 us | 2717.8 us | 1.68x | 1.72x | 1.83x | 1.02x | 0.91 | pass | open_ended |
| l1_035_groupnorm | 1650.9 us | 1145.9 us | 1.44x | 1.62x | 2.19x | 1.44x | 0.93 | kernelbench skipped: out of memory | fusion, memory_access, autotuning |
| l1_049_max_reduction_over_a_dimension | 3502.1 us | 2459.6 us | 1.42x | 1.42x | 1.16x | 1.24x | 0.99 | pass | persistent_kernel, memory_access |
| l2_013_convtranspose3d_mean_add_softmax_tanh_scaling | 830.3 us | 170.0 us | 4.88x | 5.06x | 23.97x | 19.98x | 5.87 | pass | algorithmic, dtype_fix, autotuning |
| l2_023_conv3d_groupnorm_mean | 1119.3 us | 272.4 us | 4.11x | 4.17x | 9.55x | 7.09x | 0.31 | pass | algorithmic |
| l2_045_gemm_sigmoid_logsumexp | 4749.3 us | 3891.3 us | 1.22x | 1.30x | 1.04x | 1.01x | 0.91 | pass | open_ended |
| l2_057_conv2d_relu_hardswish | 225.2 us | 178.2 us | 1.26x | 1.28x | 14.82x | 2.87x | 0.93 | pass | memory_access |
| l2_067_conv2d_gelu_globalavgpool | 1512.5 us | 397.3 us | 3.81x | 3.55x | 7.07x | 1.84x | 0.41 | pass | algorithmic |
| l2_088_gemm_groupnorm_swish_multiply_swish | 1362.0 us | 1200.2 us | 1.13x | 1.13x | 1.24x | 1.05x | 0.99 | pass | fusion, persistent_kernel |
| l2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1822.8 us | 458.7 us | 3.97x | 4.00x | 6.46x | 1.49x | 0.64 | pass | open_ended, tensor_descriptors, autotuning |
| l2_095_matmul_add_swish_tanh_gelu_hardtanh | 1465.3 us | 1218.6 us | 1.20x | 1.20x | 1.09x | 1.03x | 0.97 | pass | tensor_descriptors, persistent_kernel, autotuning |

### abl-no-rollback

| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile | Roofline | Held-out checks | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| inf_paged_attention_decode | 1196.1 us | 300.9 us | 3.98x | 4.43x | 28.92x | 13.85x | 1.01 | pass | algorithmic, persistent_kernel, autotuning |
| l1_008_matmul_with_irregular_shapes | 4584.3 us | 2604.1 us | 1.76x | 1.84x | 1.91x | 1.06x | 0.95 | pass | open_ended, tensor_descriptors, autotuning |
| l1_035_groupnorm | 1644.5 us | 1103.6 us | 1.49x | 1.49x | 2.27x | 1.50x | 0.96 | kernelbench skipped: out of memory | fusion, memory_access, autotuning |
| l1_049_max_reduction_over_a_dimension | 3804.9 us | 2682.3 us | 1.42x | 1.34x | 1.07x | 1.13x | 0.90 | pass | persistent_kernel, autotuning |
| l2_013_convtranspose3d_mean_add_softmax_tanh_scaling | 829.5 us | 135.2 us | 6.13x | 6.53x | 30.14x | 25.12x | 7.38 | pass | algorithmic, dtype_fix, autotuning |
| l2_023_conv3d_groupnorm_mean | 1119.3 us | 221.1 us | 5.06x | 5.54x | 11.77x | 8.73x | 0.38 | pass | algorithmic, dtype_fix, autotuning |
| l2_045_gemm_sigmoid_logsumexp | 4530.1 us | 3459.8 us | 1.31x | 1.34x | 1.16x | 1.14x | 1.03 | pass | memory_access, open_ended, autotuning |
| l2_057_conv2d_relu_hardswish | 225.2 us | 180.2 us | 1.25x | 1.23x | 14.65x | 2.84x | 0.92 | pass | open_ended, memory_access, autotuning |
| l2_067_conv2d_gelu_globalavgpool | 1511.5 us | 491.6 us | 3.07x | 3.01x | 5.71x | 1.49x | 0.33 | pass | algorithmic, fusion, autotuning |
| l2_088_gemm_groupnorm_swish_multiply_swish | 1364.2 us | 1165.4 us | 1.17x | 1.18x | 1.27x | 1.08x | 1.01 | pass | fusion, persistent_kernel, autotuning |
| l2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1827.8 us | 478.7 us | 3.82x | 3.81x | 6.19x | 1.43x | 0.62 | pass | open_ended, tensor_descriptors, autotuning |
| l2_095_matmul_add_swish_tanh_gelu_hardtanh | 1464.3 us | 1186.9 us | 1.23x | 1.23x | 1.11x | 1.06x | 1.00 | pass | memory_access, persistent_kernel, autotuning |

### abl-t1

| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile | Roofline | Held-out checks | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| inf_paged_attention_decode | 1194.3 us | 303.1 us | 3.94x | 4.26x | 28.71x | 13.75x | 1.00 | pass | algorithmic |
| l1_008_matmul_with_irregular_shapes | 4579.3 us | 3154.9 us | 1.45x | 1.53x | 1.58x | 0.88x | 0.78 | pass | device_specific, autotuning |
| l1_035_groupnorm | 1650.8 us | 1212.4 us | 1.36x | 1.60x | 2.07x | 1.36x | 0.88 | kernelbench skipped: out of memory | open_ended |
| l1_049_max_reduction_over_a_dimension | 3488.2 us | 2401.3 us | 1.45x | 1.43x | 1.19x | 1.27x | 1.01 | pass | open_ended |
| l2_013_convtranspose3d_mean_add_softmax_tanh_scaling | 829.5 us | 184.2 us | 4.50x | 4.71x | 22.13x | 18.45x | 5.42 | pass | algorithmic, autotuning |
| l2_023_conv3d_groupnorm_mean | 1121.3 us | 264.2 us | 4.24x | 4.30x | 9.85x | 7.31x | 0.32 | pass | algorithmic, autotuning |
| l2_045_gemm_sigmoid_logsumexp | 4786.8 us | 3801.4 us | 1.26x | 1.34x | 1.06x | 1.03x | 0.93 | pass | open_ended, persistent_kernel, autotuning |
| l2_057_conv2d_relu_hardswish | 225.3 us | 217.1 us | 1.04x | 1.07x | 12.17x | 2.36x | 0.76 | pass | autotuning |
| l2_067_conv2d_gelu_globalavgpool | 1514.9 us | 436.5 us | 3.47x | 3.84x | 6.43x | 1.68x | 0.38 | pass | algorithmic, autotuning |
| l2_088_gemm_groupnorm_swish_multiply_swish | 1365.0 us | 1173.6 us | 1.16x | 1.17x | 1.26x | 1.08x | 1.01 | pass | fusion, persistent_kernel |
| l2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1824.8 us | 555.0 us | 3.29x | 3.31x | 5.34x | 1.23x | 0.53 | pass | open_ended, tensor_descriptors, autotuning |
| l2_095_matmul_add_swish_tanh_gelu_hardtanh | 1464.3 us | 1251.4 us | 1.17x | 1.17x | 1.06x | 1.01x | 0.95 | pass | persistent_kernel, device_specific, autotuning |

### abl-t3

| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile | Roofline | Held-out checks | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| inf_paged_attention_decode | 1194.0 us | 303.1 us | 3.94x | 4.32x | 28.71x | 13.75x | 1.00 | pass | algorithmic |
| l1_008_matmul_with_irregular_shapes | 4587.7 us | 2616.6 us | 1.75x | 1.70x | 1.90x | 1.06x | 0.94 | pass | tensor_descriptors, device_specific |
| l1_035_groupnorm | 1648.7 us | 1041.6 us | 1.58x | 1.75x | 2.41x | 1.59x | 1.02 | kernelbench skipped: out of memory | open_ended |
| l1_049_max_reduction_over_a_dimension | 3500.9 us | 2377.7 us | 1.47x | 1.42x | 1.20x | 1.28x | 1.02 | pass | persistent_kernel, memory_access |
| l2_013_convtranspose3d_mean_add_softmax_tanh_scaling | 828.8 us | 141.3 us | 5.87x | 6.19x | 28.85x | 24.05x | 7.06 | pass | algorithmic, dtype_fix, autotuning |
| l2_023_conv3d_groupnorm_mean | 1118.3 us | 233.4 us | 4.79x | 4.90x | 11.15x | 8.27x | 0.36 | pass | algorithmic, dtype_fix, autotuning |
| l2_045_gemm_sigmoid_logsumexp | 4565.5 us | 3569.7 us | 1.28x | 1.20x | 1.13x | 1.10x | 0.99 | pass | open_ended, fusion |
| l2_057_conv2d_relu_hardswish | 225.2 us | 178.1 us | 1.26x | 1.34x | 14.82x | 2.87x | 0.93 | pass | open_ended, device_specific |
| l2_067_conv2d_gelu_globalavgpool | 1514.6 us | 287.5 us | 5.27x | 5.14x | 9.77x | 2.55x | 0.57 | pass | algorithmic, open_ended, autotuning |
| l2_088_gemm_groupnorm_swish_multiply_swish | 1364.0 us | 1188.2 us | 1.15x | 1.16x | 1.25x | 1.06x | 1.00 | pass | fusion, persistent_kernel |
| l2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1825.2 us | 532.5 us | 3.43x | 3.46x | 5.56x | 1.28x | 0.56 | pass | open_ended, autotuning |
| l2_095_matmul_add_swish_tanh_gelu_hardtanh | 1466.4 us | 1181.9 us | 1.24x | 1.17x | 1.12x | 1.07x | 1.00 | pass | memory_access, persistent_kernel, autotuning |

### abl-t5

| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile | Roofline | Held-out checks | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| inf_paged_attention_decode | 1195.0 us | 301.1 us | 3.97x | 4.32x | 28.90x | 13.84x | 1.01 | pass | algorithmic |
| l1_008_matmul_with_irregular_shapes | 4580.4 us | 2626.2 us | 1.74x | 1.64x | 1.89x | 1.05x | 0.94 | pass | open_ended, tensor_descriptors, autotuning |
| l1_035_groupnorm | 1646.8 us | 1166.5 us | 1.41x | 1.60x | 2.15x | 1.42x | 0.91 | kernelbench skipped: out of memory | fusion, memory_access, autotuning |
| l1_049_max_reduction_over_a_dimension | 3498.0 us | 2589.7 us | 1.35x | 1.26x | 1.10x | 1.18x | 0.94 | pass | persistent_kernel, memory_access |
| l2_013_convtranspose3d_mean_add_softmax_tanh_scaling | 829.1 us | 145.4 us | 5.70x | 5.80x | 28.02x | 23.36x | 6.86 | pass | algorithmic, dtype_fix, autotuning |
| l2_023_conv3d_groupnorm_mean | 1121.0 us | 221.2 us | 5.07x | 5.28x | 11.76x | 8.73x | 0.38 | pass | algorithmic, memory_access, autotuning |
| l2_045_gemm_sigmoid_logsumexp | 4823.4 us | 3778.6 us | 1.28x | 1.32x | 1.07x | 1.04x | 0.94 | pass | algorithmic, open_ended |
| l2_057_conv2d_relu_hardswish | 225.2 us | 219.1 us | 1.03x | 1.04x | 12.05x | 2.34x | 0.76 | pass | open_ended |
| l2_067_conv2d_gelu_globalavgpool | 1514.5 us | 344.1 us | 4.40x | 4.26x | 8.16x | 2.13x | 0.48 | pass | algorithmic, fusion, autotuning |
| l2_088_gemm_groupnorm_swish_multiply_swish | 1366.0 us | 1210.5 us | 1.13x | 1.13x | 1.22x | 1.04x | 0.98 | pass | fusion, persistent_kernel |
| l2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1828.9 us | 423.9 us | 4.31x | 4.28x | 6.99x | 1.61x | 0.70 | pass | open_ended, tensor_descriptors, autotuning |
| l2_095_matmul_add_swish_tanh_gelu_hardtanh | 1468.4 us | 1184.8 us | 1.24x | 1.24x | 1.12x | 1.06x | 1.00 | pass | persistent_kernel, memory_access, autotuning |

### abl-best3

| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile | Roofline | Held-out checks | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| inf_paged_attention_decode | 1191.9 us | 299.0 us | 3.99x | 4.41x | 29.10x | 13.94x | 1.02 | pass | algorithmic, persistent_kernel |
| l1_008_matmul_with_irregular_shapes | 4577.3 us | 2657.3 us | 1.72x | 1.71x | 1.87x | 1.04x | 0.93 | pass | open_ended, device_specific |
| l1_035_groupnorm | 1649.7 us | 1039.6 us | 1.59x | 1.60x | 2.41x | 1.59x | 1.02 | kernelbench skipped: out of memory | open_ended, autotuning |
| l1_049_max_reduction_over_a_dimension | 3503.1 us | 2394.4 us | 1.46x | 1.47x | 1.19x | 1.27x | 1.01 | pass | open_ended |
| l2_013_convtranspose3d_mean_add_softmax_tanh_scaling | 829.5 us | 121.9 us | 6.81x | 7.09x | 33.44x | 27.88x | 8.19 | pass | algorithmic, open_ended, autotuning |
| l2_023_conv3d_groupnorm_mean | 1119.5 us | 184.3 us | 6.07x | 6.14x | 14.12x | 10.48x | 0.46 | pass | algorithmic, memory_access, autotuning |
| l2_045_gemm_sigmoid_logsumexp | 4659.2 us | 3525.2 us | 1.32x | 1.21x | 1.14x | 1.12x | 1.01 | pass | open_ended, fusion |
| l2_057_conv2d_relu_hardswish | 225.2 us | 188.4 us | 1.20x | 1.21x | 14.02x | 2.72x | 0.88 | pass | memory_access, autotuning |
| l2_067_conv2d_gelu_globalavgpool | 1510.6 us | 365.6 us | 4.13x | 4.52x | 7.68x | 2.00x | 0.45 | pass | algorithmic, open_ended, autotuning |
| l2_088_gemm_groupnorm_swish_multiply_swish | 1362.9 us | 1165.3 us | 1.17x | 1.17x | 1.27x | 1.08x | 1.01 | pass | fusion, persistent_kernel |
| l2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1827.2 us | 520.3 us | 3.51x | 3.51x | 5.69x | 1.32x | 0.57 | pass | open_ended, device_specific, autotuning |
| l2_095_matmul_add_swish_tanh_gelu_hardtanh | 1463.3 us | 1190.9 us | 1.23x | 1.24x | 1.11x | 1.06x | 0.99 | pass | persistent_kernel, device_specific, autotuning |

### abl-opus-low

| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile | Roofline | Held-out checks | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| inf_paged_attention_decode | 1199.2 us | 301.4 us | 3.98x | 4.33x | 28.87x | 13.83x | 1.01 | pass | algorithmic |
| l1_008_matmul_with_irregular_shapes | 4880.4 us | 2760.3 us | 1.77x | 1.60x | 1.80x | 1.00x | 0.89 | pass | tensor_descriptors |
| l1_035_groupnorm | 1648.7 us | 1039.4 us | 1.59x | 1.78x | 2.41x | 1.59x | 1.02 | kernelbench skipped: out of memory | algorithmic |
| l1_049_max_reduction_over_a_dimension | 3493.3 us | 2521.8 us | 1.39x | 1.37x | 1.13x | 1.21x | 0.96 | pass | persistent_kernel, memory_access, autotuning |
| l2_013_convtranspose3d_mean_add_softmax_tanh_scaling | 829.5 us | 153.6 us | 5.40x | 5.60x | 26.54x | 22.12x | 6.50 | pass | algorithmic, autotuning |
| l2_023_conv3d_groupnorm_mean | 1120.3 us | 366.7 us | 3.06x | 2.97x | 7.10x | 5.27x | 0.23 | pass | algorithmic, autotuning |
| l2_045_gemm_sigmoid_logsumexp | 4500.2 us | 3633.9 us | 1.24x | 1.29x | 1.11x | 1.08x | 0.98 | pass | open_ended |
| l2_057_conv2d_relu_hardswish | 225.2 us | 208.9 us | 1.08x | 1.08x | 12.64x | 2.45x | 0.79 | pass | open_ended |
| l2_067_conv2d_gelu_globalavgpool | 1519.6 us | 542.8 us | 2.80x | 2.41x | 5.18x | 1.35x | 0.30 | pass | algorithmic, dtype_fix |
| l2_088_gemm_groupnorm_swish_multiply_swish | 1364.0 us | 1177.5 us | 1.16x | 1.17x | 1.26x | 1.07x | 1.00 | pass | fusion, persistent_kernel |
| l2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1826.9 us | 1186.8 us | 1.54x | 1.71x | 2.50x | 0.58x | 0.25 | pass | open_ended, tensor_descriptors |
| l2_095_matmul_add_swish_tanh_gelu_hardtanh | 1465.3 us | 1251.4 us | 1.17x | 1.18x | 1.06x | 1.01x | 0.95 | pass | tensor_descriptors, persistent_kernel |

### abl-opus-high

| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile | Roofline | Held-out checks | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| inf_paged_attention_decode | 1195.0 us | 303.1 us | 3.94x | 4.45x | 28.71x | 13.75x | 1.00 | pass | algorithmic |
| l1_008_matmul_with_irregular_shapes | 4580.4 us | 2583.6 us | 1.77x | 1.83x | 1.93x | 1.07x | 0.95 | pass | open_ended, autotuning |
| l1_035_groupnorm | 1652.4 us | 1077.5 us | 1.53x | 1.68x | 2.33x | 1.53x | 0.98 | kernelbench skipped: out of memory | fusion, memory_access, autotuning |
| l1_049_max_reduction_over_a_dimension | 3526.7 us | 2402.3 us | 1.47x | 1.42x | 1.19x | 1.27x | 1.01 | pass | persistent_kernel, open_ended |
| l2_013_convtranspose3d_mean_add_softmax_tanh_scaling | 829.5 us | 122.9 us | 6.75x | 7.16x | 33.16x | 27.64x | 8.12 | pass | algorithmic, dtype_fix |
| l2_023_conv3d_groupnorm_mean | 1125.1 us | 196.6 us | 5.72x | 6.10x | 13.23x | 9.82x | 0.43 | pass | algorithmic, open_ended, autotuning |
| l2_045_gemm_sigmoid_logsumexp | 4732.6 us | 3792.9 us | 1.25x | 1.24x | 1.06x | 1.04x | 0.94 | pass | open_ended, fusion |
| l2_057_conv2d_relu_hardswish | 225.3 us | 217.0 us | 1.04x | 1.05x | 12.17x | 2.36x | 0.76 | pass | autotuning |
| l2_067_conv2d_gelu_globalavgpool | 1511.7 us | 286.8 us | 5.27x | 5.20x | 9.80x | 2.55x | 0.57 | pass | algorithmic, open_ended |
| l2_088_gemm_groupnorm_swish_multiply_swish | 1364.0 us | 1173.3 us | 1.16x | 1.17x | 1.26x | 1.08x | 1.01 | pass | fusion, persistent_kernel |
| l2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1823.7 us | 455.7 us | 4.00x | 4.02x | 6.50x | 1.50x | 0.65 | pass | open_ended, tensor_descriptors, autotuning |
| l2_095_matmul_add_swish_tanh_gelu_hardtanh | 1464.3 us | 1191.3 us | 1.23x | 1.28x | 1.11x | 1.06x | 0.99 | pass | persistent_kernel, device_specific |

### abl-sonnet

| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile | Roofline | Held-out checks | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| inf_paged_attention_decode | 1196.0 us | 303.1 us | 3.95x | 4.59x | 28.71x | 13.75x | 1.00 | pass | algorithmic |
| l1_008_matmul_with_irregular_shapes | 4946.0 us | 2980.3 us | 1.66x | 1.62x | 1.67x | 0.93x | 0.83 | pass | open_ended, autotuning |
| l1_035_groupnorm | 1651.7 us | 1039.2 us | 1.59x | 1.91x | 2.42x | 1.59x | 1.02 | kernelbench skipped: out of memory | open_ended |
| l1_049_max_reduction_over_a_dimension | 3511.5 us | 2522.9 us | 1.39x | 1.43x | 1.13x | 1.21x | 0.96 | pass | open_ended, autotuning |
| l2_013_convtranspose3d_mean_add_softmax_tanh_scaling | 830.4 us | 122.9 us | 6.75x | 7.10x | 33.15x | 27.64x | 8.12 | pass | algorithmic, dtype_fix, autotuning |
| l2_023_conv3d_groupnorm_mean | 1121.1 us | 266.2 us | 4.21x | 3.91x | 9.77x | 7.25x | 0.32 | pass | algorithmic, autotuning |
| l2_045_gemm_sigmoid_logsumexp | 4445.2 us | 3602.5 us | 1.23x | 1.19x | 1.12x | 1.09x | 0.98 | pass | open_ended, autotuning |
| l2_057_conv2d_relu_hardswish | 225.2 us | 217.0 us | 1.04x | 1.05x | 12.17x | 2.36x | 0.76 | pass | autotuning |
| l2_067_conv2d_gelu_globalavgpool | 1524.8 us | 518.1 us | 2.94x | 2.78x | 5.42x | 1.41x | 0.32 | pass | algorithmic, fusion, autotuning |
| l2_088_gemm_groupnorm_swish_multiply_swish | 1366.0 us | 1192.9 us | 1.15x | 1.15x | 1.24x | 1.06x | 0.99 | pass | fusion, persistent_kernel, autotuning |
| l2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1825.8 us | 1217.8 us | 1.50x | 1.52x | 2.43x | 0.56x | 0.24 | pass | algorithmic |
| l2_095_matmul_add_swish_tanh_gelu_hardtanh | 1467.4 us | 1215.4 us | 1.21x | 1.20x | 1.09x | 1.04x | 0.97 | pass | memory_access, persistent_kernel, autotuning |

### abl-haiku

| Problem | Seed | Final | Over seed | Kernel only | Over eager | Over compile | Roofline | Held-out checks | Accepted |
|---|---|---|---|---|---|---|---|---|---|
| inf_paged_attention_decode | 1194.8 us | 301.1 us | 3.97x | 4.31x | 28.90x | 13.84x | 1.01 | pass | algorithmic, persistent_kernel |
| l1_008_matmul_with_irregular_shapes | 4581.4 us | 2730.6 us | 1.68x | 1.67x | 1.82x | 1.01x | 0.90 | pass | memory_access |
| l1_035_groupnorm | 1650.7 us | 1244.3 us | 1.33x | 1.32x | 2.02x | 1.33x | 0.85 | kernelbench skipped: out of memory | open_ended, memory_access, autotuning |
| l1_049_max_reduction_over_a_dimension | 3479.6 us | 2483.4 us | 1.40x | 1.37x | 1.15x | 1.23x | 0.98 | pass | open_ended, memory_access, autotuning |
| l2_013_convtranspose3d_mean_add_softmax_tanh_scaling | 829.9 us | 155.7 us | 5.33x | 5.35x | 26.18x | 21.82x | 6.41 | pass | algorithmic, autotuning |
| l2_023_conv3d_groupnorm_mean | 1120.3 us | 268.3 us | 4.18x | 4.17x | 9.70x | 7.20x | 0.32 | pass | algorithmic, autotuning |
| l2_045_gemm_sigmoid_logsumexp | 4374.6 us | 3653.7 us | 1.20x | 1.14x | 1.10x | 1.08x | 0.97 | pass | open_ended |
| l2_057_conv2d_relu_hardswish | 225.3 us | 216.9 us | 1.04x | 1.05x | 12.17x | 2.36x | 0.76 | pass | autotuning |
| l2_067_conv2d_gelu_globalavgpool | 1511.5 us | 608.2 us | 2.49x | 2.52x | 4.62x | 1.20x | 0.27 | pass | algorithmic |
| l2_088_gemm_groupnorm_swish_multiply_swish | 1364.0 us | 1236.0 us | 1.10x | 1.11x | 1.20x | 1.02x | 0.96 | pass | fusion, autotuning |
| l2_091_convtranspose2d_softmax_biasadd_scaling_sigmoid | 1826.8 us | 1214.4 us | 1.50x | 1.50x | 2.44x | 0.56x | 0.24 | pass | open_ended, memory_access |
| l2_095_matmul_add_swish_tanh_gelu_hardtanh | 1463.3 us | 1347.6 us | 1.09x | 1.09x | 0.98x | 0.93x | 0.88 | pass | memory_access, persistent_kernel |
