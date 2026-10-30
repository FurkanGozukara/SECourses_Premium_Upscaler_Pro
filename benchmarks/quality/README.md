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
