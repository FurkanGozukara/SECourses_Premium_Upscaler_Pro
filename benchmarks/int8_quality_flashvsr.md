# INT8 ConvRot quality benchmark - flashvsr

- source: `ComfyUI-FlashVSR_Stable\models\FlashVSR-v1.1\diffusion_pytorch_model_streaming_dmd.safetensors`
- calibration: no, hessians (GPTQ): no
- features: `{'clip_search': 'mse-0.55-1.0-80', 'ls_refit': True, 'qmin': -128, 'scale_groups': 'row', 'lowrank_rank': 16, 'rescue_budget_mb': 256.0, 'gptq': False, 'calibrated': False}`

## Model-wide (signal base = union of quantized layers)

| config | quantized layers | weight SQNR | est. quantized-weights size |
|---|---|---|---|
| V6.0 legacy | 305 | 41.67 dB | 1.42 GB |
| V6.1 new | 292 (+8 rescued) | 42.16 dB | 1.80 GB |

## Worst 10 layers under the new pipeline

| layer | params | V6.0 SQNR | V6.1 SQNR | V6.1 features |
|---|---|---|---|---|
| blocks.11.ffn.2 | 13.8M | 41.02 dB | 41.14 dB | ara  |
| blocks.9.ffn.2 | 13.8M | 41.02 dB | 41.15 dB | ara  |
| blocks.23.ffn.2 | 13.8M | 41.02 dB | 41.15 dB | ara  |
| blocks.20.ffn.2 | 13.8M | 41.03 dB | 41.15 dB | ara  |
| blocks.8.ffn.2 | 13.8M | 41.03 dB | 41.16 dB | ara  |
| blocks.13.ffn.2 | 13.8M | 41.03 dB | 41.16 dB | ara  |
| blocks.21.ffn.2 | 13.8M | 41.03 dB | 41.16 dB | ara  |
| blocks.26.ffn.2 | 13.8M | 41.03 dB | 41.16 dB | ara  |
| blocks.17.ffn.2 | 13.8M | 41.03 dB | 41.16 dB | ara  |
| blocks.7.ffn.2 | 13.8M | 41.03 dB | 41.16 dB | ara  |

Layers quantized by both where V6.1 wins on SQNR: 292/292

Benchmark wall time: 15.5s