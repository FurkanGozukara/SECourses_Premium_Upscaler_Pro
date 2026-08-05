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
