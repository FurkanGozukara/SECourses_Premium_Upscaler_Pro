# INT8 ConvRot quality benchmark - seedvr2

- source: `G:\SECourses_Premium_Upscaler_Pro_V6\SECourses_Premium_Upscaler_Pro\SeedVR2\models\seedvr2_ema_7b_sharp_fp16.safetensors`
- calibration: no, hessians (GPTQ): no
- features: `{'clip_search': 'mse-0.55-1.0-80', 'ls_refit': True, 'qmin': -128, 'scale_groups': 'row', 'lowrank_rank': 16, 'rescue_budget_mb': 256.0, 'gptq': False, 'calibrated': False}`

## Model-wide (signal base = union of quantized layers)

| config | quantized layers | weight SQNR | est. quantized-weights size |
|---|---|---|---|
| V6.0 legacy | 288 | 41.28 dB | 8.33 GB |
| V6.1 new | 260 (+28 rescued) | 41.53 dB | 8.70 GB |

## Worst 10 layers under the new pipeline

| layer | params | V6.0 SQNR | V6.1 SQNR | V6.1 features |
|---|---|---|---|---|
| blocks.3.attn.proj_out.vid | 9.4M | 40.78 dB | 40.91 dB | ara  |
| blocks.24.attn.proj_out.vid | 9.4M | 40.81 dB | 40.93 dB | ara  |
| blocks.34.attn.proj_out.vid | 9.4M | 40.74 dB | 40.94 dB | ara  |
| blocks.7.attn.proj_out.vid | 9.4M | 40.80 dB | 40.94 dB | ara  |
| blocks.4.attn.proj_out.vid | 9.4M | 40.83 dB | 40.95 dB | ara  |
| blocks.15.mlp.vid.proj_out | 37.7M | 40.88 dB | 40.97 dB | ara  |
| blocks.18.mlp.vid.proj_out | 37.7M | 40.88 dB | 40.97 dB | ara  |
| blocks.11.mlp.vid.proj_out | 37.7M | 40.88 dB | 40.97 dB | ara  |
| blocks.1.mlp.vid.proj_out | 37.7M | 40.89 dB | 40.97 dB | ara  |
| blocks.25.mlp.vid.proj_out | 37.7M | 40.88 dB | 40.97 dB | ara  |

Layers quantized by both where V6.1 wins on SQNR: 260/260

Benchmark wall time: 70.7s