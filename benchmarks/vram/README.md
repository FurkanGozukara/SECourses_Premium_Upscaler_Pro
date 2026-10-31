# VRAM calibration data for Auto Tune

Auto Tune never runs test clips: each planner in `shared/vram_autotune/` computes the settings from
memory formulas fitted to the measurements in this folder (one JSON line per real run, RTX 5090,
torch 2.13 + CUDA 13, native caching allocator, `garbage_collection_threshold:0.8`).

Each record holds, per process of the run (SeedVR2 encode / upscale / decode phases, SparkVSR stages),
the PyTorch peak reserved and allocated memory, plus the whole process tree's peak dedicated and shared
GPU memory from the Windows per-process counters. `tree_shared_growth_gb` stays at about 0.08-0.11 GB in
every record (the driver's baseline), including runs pushed past their cap: no run spilled into shared
GPU memory. Runs past the cap end with `oom: true` (a normal CUDA out-of-memory error).

| File | Planner | Fitted terms (GiB, PyTorch allocated) |
|---|---|---|
| `seedvr2.jsonl` | `seedvr2.py` | DiT 7B INT8: 8.38 + 0.176 per 1k latent tokens, BlockSwap 0.21 per block; VAE encode 3.85 + 7.1 per tile-MP, decode 3.4 + 14.0 per tile-MP |
| `flashvsr.jsonl` | `flashvsr.py` | 3.4 + 7.7 per padded output MP (or tile MP) + 6 MB per MP per frame; kv_ratio 3 to 1 saves 24% of the per-MP term |
| `sparkvsr.jsonl` | `sparkvsr.py` | transformer BF16 10.4 + 0.076 per 1k tokens, INT8 ConvRot 6.3 + 0.170 per 1k tokens, tokens = (frames/8 + 1) x (H/16) x (W/16); one process (no stage isolation) +0.9 GB |
| `ltx25.jsonl` | `LTX25/ltx25_engine/planner.py` | non-block weights 4.55 + fixed 3.0 (two streaming block buffers, tiled VAE) + 0.25 MB per sequence token (video tokens x 1.25 + 2048); blocks that do not fit stream from RAM |
| `rtx.jsonl` | `rtx.py` | whole process 0.6 + 0.06 per output MP + 0.3 per input MP (0.75 GB for 1080p output, 3.0 GB for 8K; the SDK allocates outside PyTorch) |

Planners add a safety margin (8-20% plus 0.6 GB) because the cap limits *reserved* memory, which the
allocator keeps above allocated (1.2-1.4x). Near the cap, fragmentation decides: `seedvr2.jsonl` holds boundary runs
(`bnd_*`, `frag_*`) where a planned batch failed with 4 GB stranded in split blocks and passed with
`max_split_size_mb:512`, which the launcher adds when a plan uses more than 80% of its budget (`speed_*`: it
costs about 10-15% speed, so roomy plans skip it).

Re-measure after an engine or model change:

```
set CALIB_GPU_LUID=<GPU 0 LUID from tools\gpu_process_memory.py>
venv\Scripts\python.exe tools\vram_calibrate.py seedvr2 --cases my_cases.json
```

A case is a JSON object with `name`, the source size (`in_w`, `in_h`, `frames`) and the engine settings
(see `tools/vram_calibrate.py`); `target_cap_gb` sets Save VRAM so the cap lands on that value (boundary
test). The tool takes the shared GPU 0 turn lock and kills a case whose shared GPU memory grows by more
than 1 GB.
