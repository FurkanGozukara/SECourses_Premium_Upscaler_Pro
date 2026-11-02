# INT8 ConvRot V2 (V6.1) — pipeline, calibration and shipping guide

## What the converter does now

For every eligible DiT Linear layer (SparkVSR / FlashVSR+ / SeedVR2):

1. **Layer policy** (`shared/int8_layer_policy.py`) — generic rules replace the old
   per-model lists: only Linears inside the repeated transformer-block stack are
   quantized; conditioning layers (timestep/text embedders, modulation/adaLN,
   input/output projections) and tiny GEMMs stay in source precision.
2. **Outlier clamp** — |w| > 1000 is zeroed before rotation (corrupted-checkpoint
   insurance).
3. **Hadamard rotation** — unchanged regular ConvRot rotation, group 256/64/16.
4. **Scale search** — MSE clip search (80 steps, 0.55–1.0×absmax) followed by a
   closed-form least-squares refit; codes use the full [-128, 127] range.
   With calibration stats the search is activation-energy weighted.
5. **GPTQ rounding** — only when a calibration Hessian file exists.
6. **ARA low-rank recovery** — rank-16 (default) correction of the residual,
   stored as `<layer>.int8_ara_down/up` bf16 tensors in the same cache file and
   applied by the runtime as one skinny extra GEMM. Hessian- or energy-weighted
   when calibration exists, plain SVD otherwise.
7. **Bias correction** — with calibration stats, the mean output shift of the
   quantization error is folded into the layer bias.
8. **Rescue budget** — the worst-quantizing layers (by relative error) are kept in
   source precision under a byte budget (default 256 MB).

Everything lands in the **same single safetensors cache file** as before; the
runtime (`int8_convrot_linear`) reads per-row or per-group scales and the ARA
tensors transparently. Old V6.0 caches carry an older format marker and are
rebuilt automatically once.

## Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `SECOURSES_INT8_LOWRANK_RANK` | `16` | ARA rank; `0` disables ARA |
| `SECOURSES_INT8_RESCUE_MB` | `256` | Rescue budget in MB; `0` disables |
| `SECOURSES_INT8_SCALE_GROUPS` | `row` | `rot` = per-(row, rotation-group) weight scales (needs Triton at runtime for the fast path; otherwise falls back to the dequantized matmul) |
| `SECOURSES_INT8_GPTQ` | `1` | `0` disables GPTQ even when Hessians exist |
| `SECOURSES_INT8_CALIBRATE` | off | `1` during a BF16/FP16 run records calibration stats |
| `SECOURSES_INT8_HESSIAN` | off | `1` additionally records full Hessians (large, local-only) |
| `SECOURSES_INT8_HESSIAN_MAX_IN` | `6144` | skip Hessians for layers wider than this |

## Maximum-quality cache generation (what we do before shipping)

1. **Calibrate** — run one representative upscale per model on the plain
   BF16/FP16 model with:

   ```
   set SECOURSES_INT8_CALIBRATE=1
   set SECOURSES_INT8_HESSIAN=1
   ```

   This writes `<checkpoint>.int8_calib.safetensors` (small) and
   `<checkpoint>.int8_hessian.safetensors` (several GB, stays on your machine)
   next to the source model when the app process exits.

2. **Generate** — unset the calibrate vars, select the INT8 option in the app
   (or delete the old cache) and let the converter run once. It automatically
   picks up the calibration artifacts and bakes GPTQ + energy weighting +
   Hessian-ARA + bias correction into the cache.

3. **Benchmark** — `python tools/int8_quality_benchmark.py --model <name>`
   compares V6.0 vs V6.1 weight SQNR; reports land in `benchmarks/`.

4. **Ship** — upload the generated `*_int8_convrot.safetensors` /
   `SparkVSR-int8-convrot.safetensors`. Cache validation is content-based
   (format marker + source file size), so the file loads on any machine.
   Users who do not download it still get automatic on-first-use generation
   (data-free mode: everything except GPTQ / energy weighting / bias
   correction, which need the calibration run).

## Runtime notes

- The fused Triton activation-quantize kernel now computes in fp32 (the bf16
  division previously added up to ±0.5 LSB of noise on the largest values).
- ARA adds one `M×in×rank` + `M×rank×out` bf16 GEMM per layer (~1% cost).
- Per-group scales use a dedicated Triton kernel; without Triton the layer
  falls back to the dequantized matmul path (identical output, BF16 speed,
  still half weight VRAM). This is why per-group is opt-in.

## Speed audit 2026-11-02

RTX 5090, idle GPU, app command lines (ski 320x180 61 f -> 720p, robot 480x270 49 f -> 1080p).

| Model | BF16 | INT8 ConvRot | Note |
|---|---|---|---|
| FlashVSR v1.1, robot 1080p (total) | 15.3 s | 14.7-14.9 s | Linears ~17 % of GPU time; attention, convs, copies dominate |
| FlashVSR+ Turbo, robot 1080p (total, after RoPE fix) | 11.3-12.3 s | 11.1 s | INT8 earlier "40 % slower" was the old FP32 path |
| Open-H3 FL2VA Pruned | does not fit 32 GB | refine 14.1 s (3 steps, 35k tokens) | all Linears on comfy_kitchen int8_linear, LoRA merged into all 208 weights |

Kernel level (real layer shapes, 4k/16k tokens, with ARA): SeedVR2 3B 2.0x, SparkVSR 2.1-2.5x, Open-H3 2.9-3.0x (shapes through shared/int8_convrot), FlashVSR 1.0-1.5x (1536-wide layers: the ARA branch costs about as much as the INT8 GEMM).
An FP32 input (no autocast cast) drops every model to 0.6-2.0x.

Changes: FlashVSR+ RoPE uses comfy_kitchen's fused `apply_rope1` instead of float64 math (`flashvsr_plus_engine/models/embeddings.py`, Turbo 12-15 % faster, +0.001 dB vs GT); `_ara_down_unrotated` is cached per tensor (bit-identical output).
The same fused RoPE on FlashVSR v1.1 gave no measurable gain and was reverted.

Harness (session scratchpad, rebuild when needed): `prof_wrap.py` runs a CLI under torch.profiler and sums CUDA kernel time by category;
`f64_wrap.py` is a TorchDispatchMode that logs the call site of every CUDA op producing float64; `h3_wrap.py` counts comfy_kitchen ops and QuantizedTensor dequantize calls in an Open-H3 run.

W4A8 / W6A8 (comfy_kitchen `w4a8_int8_linear`, same day, real layer shapes): the same INT8 tensor-core GEMM as INT8 ConvRot, so no faster at DiT token counts
(Turbo 1080p linears: INT8 2.01x, W4A8 1.97x, W6A8 1.98x of BF16; Open-H3 35k tokens: 3.70x / 3.72x / 3.86x). Their gain is VRAM only (worth it where the INT8 DiT does not fit, e.g. Open-H3 on 24 GB).
FlashVSR+ INT8 caches are built without ARA since cache format v2 (`plus_runtime.build_int8_cache`): Turbo robot 1080p 11.0-12.0 s vs 11.4-12.7 s with ARA and 11.4-11.6 s BF16,
GT 37.248 / 37.245 / 37.242 dB, ski 32.816 / 32.813 / 32.816 dB. FlashVSR+ follows the Precision setting (the forced-BF16 override is gone; Windows default `int8_convrot`); Auto Tune plans its INT8 like BF16.
