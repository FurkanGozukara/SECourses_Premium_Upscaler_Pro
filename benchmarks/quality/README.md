# Quality vs batch / chunk length

Real 720p footage (and a 4:3 clip, `q_43_*`) was downscaled by each model's factor, upscaled back and
compared with the original frames: PSNR, SSIM and a temporal error (how far the output's frame-to-frame
change differs from the original's: flicker and wrong motion). Only the batch / chunk length changed
between runs of a model.

| Model | Result | Planner rule |
|---|---|---|
| SeedVR2 7B INT8, 121 frames | batch 121 best on every score (+0.32 dB, -4 % temporal error vs batch 21) | largest batch that fits (window RoPE: any length is in range) |
| SparkVSR INT8, 129 frames, two clips | 33-frame chunks best on both (+0.15 dB, lower temporal error); longer chunks only faster | 33-frame chunks |
| LTX 2.5, 481 frames | within 20 s: shorter chunks slightly higher PSNR, longer chunks less flicker | longest chunk up to 20 s |
| LTX 2.5, 721 frames | one 24 s chunk below 20 s chunks in every segment (positions are seconds / 20) | chunks stop at 20 s |
| FlashVSR+, 4,200 frames | one pass crashes (temporal RoPE table of 1024 positions); 4,000-frame passes work | passes of at most 4,000 frames |

`temporal_limit_results.jsonl` holds the LTX 24 s / 20 s, SparkVSR 129 / 81 and FlashVSR 4,200-frame runs.

# Decoders, precision, sparse attention and audio conditioning (V9.1, 31 October 2026)

Same method (downscale, upscale back, compare with the original), RTX 5090.

| Model | Compared | Result | Default |
|---|---|---|---|
| FlashVSR+ 1.1, robot 720p + talking head 832x1248, 81 frames | TC decoder (tiny) / full + Wan2.1 VAE / full + LightVAE_W2.1 | LightVAE best on both clips: 27.97 / 27.80 dB vs TC 27.69 / 27.43 and Wan2.1 27.37 / 27.18, least flicker, detail 1.16x the original's (TC 1.36x, Wan2.1 1.41x: invented detail, thick glasses rims), fastest | full + LightVAE_W2.1 |
| FlashVSR+ 1.1 | BF16 / INT8 ConvRot | within 0.02 dB, -0.99 GB allocated | INT8 on Windows |
| Open-H3 2K, 49 frames | dense / tau 1.2 window (old) / step 1 dense + tau 1.0 / all sparse tau 1.75 | refine 44 / 32 / 34 / 23 s; 37.0 dB vs dense for step-1-dense, 32.7 and 30.3 dB for the others, whose sparse first step reshaped a robot arm | step 1 dense, tau 1.0 |
| Open-H3, talking head 416x624 -> 832x1248, 240 frames | encoded silence / source audio | source: mouth +0.65 dB (every 2 s segment), mouth-open correlation 0.87 vs 0.82; silence parted closed lips | Source soundtrack |
| Open-H3 4K | one refine from 0.8 / 0.7 then a same-size pass from 0.6 | second pass: 2.1x refine time, grain and invented objects | off |

`flashvsr_decoder_results.jsonl` (quality, quality2, vram, vram2, acc) and `open_h3_quality_results.jsonl` (sparse, lips, ladder).
