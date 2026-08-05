# INT8 ConvRot quality benchmark - sparkvsr

- source: `G:\SECourses_Premium_Upscaler_Pro_V6\SECourses_Premium_Upscaler_Pro\SparkVSR\models\SparkVSR-bf16\transformer\diffusion_pytorch_model.safetensors`
- calibration: no, hessians (GPTQ): no
- features: `{'clip_search': 'mse-0.55-1.0-80', 'ls_refit': True, 'qmin': -128, 'scale_groups': 'row', 'lowrank_rank': 16, 'rescue_budget_mb': 256.0, 'gptq': False, 'calibrated': False}`

## Model-wide (signal base = union of quantized layers)

| config | quantized layers | weight SQNR | est. quantized-weights size |
|---|---|---|---|
| V6.0 legacy | 252 | 41.35 dB | 6.38 GB |
| V6.1 new | 239 (+13 rescued) | 41.71 dB | 6.72 GB |

## Worst 10 layers under the new pipeline

| layer | params | V6.0 SQNR | V6.1 SQNR | V6.1 features |
|---|---|---|---|---|
| transformer_blocks.18.ff.net.2 | 37.7M | 40.91 dB | 41.00 dB | ara  |
| transformer_blocks.27.ff.net.2 | 37.7M | 40.90 dB | 41.00 dB | ara  |
| transformer_blocks.0.ff.net.2 | 37.7M | 40.91 dB | 41.00 dB | ara  |
| transformer_blocks.24.ff.net.2 | 37.7M | 40.91 dB | 41.00 dB | ara  |
| transformer_blocks.15.ff.net.2 | 37.7M | 40.90 dB | 41.01 dB | ara  |
| transformer_blocks.31.ff.net.2 | 37.7M | 40.86 dB | 41.01 dB | ara  |
| transformer_blocks.13.ff.net.2 | 37.7M | 40.91 dB | 41.01 dB | ara  |
| transformer_blocks.23.ff.net.2 | 37.7M | 40.92 dB | 41.01 dB | ara  |
| transformer_blocks.12.ff.net.2 | 37.7M | 40.91 dB | 41.01 dB | ara  |
| transformer_blocks.26.ff.net.2 | 37.7M | 40.92 dB | 41.01 dB | ara  |

Layers quantized by both where V6.1 wins on SQNR: 239/239

Benchmark wall time: 37.7s