# SECourses Ultimate Video and Image Upscaler Pro V6.0

## SeedVR2 • FlashVSR+ • SparkVSR • NVIDIA RTX Super Resolution • GAN Upscalers • RIFE • Face Restoration • NEW INT8 ConvRot — The Fully Automated Movie Restoration and Upscale Studio

**Download (Patreon):** https://www.patreon.com/posts/150202809

**4K Tutorial:** https://youtu.be/_WT4C78j5-c • **Quick 4K demo:** https://youtu.be/bPWsg8DREiM

---

Turn any low-resolution video or image into crisp 4K/8K — on your own GPU, with one click. Upscaler Pro is the only app that puts **all seven state-of-the-art upscaling engines** behind one polished interface, installs itself fully automatically on **Windows, RunPod, Massed Compute and SimplePod**, and is optimized so hard that a **120-minute movie upscales end-to-end on a consumer GPU** — with scene detection, resume, queueing, audio handling and color fixing all done for you.

It runs on everything from an **RTX 2000-series** card all the way to **H100 / H200 / B200 / RTX PRO 6000**, with pre-compiled Torch 2.13 + CUDA 13 wheels, working torch.compile, FlashAttention, SageAttention and Triton — zero manual setup.

---

## NEW in V6.0: INT8 ConvRot Quantization — Faster Than BF16, Better Than FP8

V6 brings the same cutting-edge quantization that recently landed in ComfyUI and our Musubi Trainer to video upscaling: **group-wise Hadamard rotation + per-row MSE-optimized INT8 weights**, executed with fused Triton INT8 tensor-core matmuls.

Measured head-to-head on an RTX 5090 (SparkVSR, 33 frames, 480×272 → 1920×1088, identical settings):

![Benchmark table](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/17_benchmark_table.png)

![Benchmark chart](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/18_benchmark_chart.png)

- **SparkVSR:** new `SparkVSR-int8-convrot` model choice — **11% faster than BF16, 26% faster than FP8 Scaled**, ~4 GB less peak VRAM than BF16, and **closer to BF16 quality than FP8 is** (49.1 dB vs 47.8 dB PSNR). The INT8 model is generated automatically from the BF16 weights the first time you pick it.
- **SeedVR2:** new **INT8 ConvRot (DiT)** checkbox — about **half the DiT weight VRAM** and ~15% faster sampling on the FP16 models, quantized on load.
- **FlashVSR+:** new `int8_convrot` precision — half the DiT weight VRAM with near-BF16 output.
- Works on **RTX 2000 (Turing) and newer**; older GPUs automatically fall back to a compatible path with identical output. No extra downloads, no extra setup.

---

## Also New in V6.0

- **Modernized interface on Gradio 6.22** — every button on every tab has its own distinct color, so you always know where you are and what you are clicking. Tab switching stays fluid even while a heavy upscale is running (measured 0.1–0.3 s under full GPU load).
- **Version History tab** — the complete changelog of the app, inside the app, each release in its own expandable section.
- **Polished light theme** (dark stays the default) and **theme choice that sticks** across restarts.
- **No more silent failures** — every invalid input now shows an instant, clear error message on every tab, and unexpected errors pop up as toasts.
- **Deep compatibility pass** on the newest Torch 2.13 + CUDA 13 + Gradio 6.22 stack: fresh-install FlashVSR+ setup fixed, RIFE fixed for the current model layout and NumPy 2, GGUF support restored, video saving fixed, SparkVSR dependencies restored.

---

## The Seven Engines

### 1. SeedVR2 — the quality king for low-resolution sources

![SeedVR2](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/01_seedvr2_dark.png)

The legendary one-step diffusion restorer. 3B and 7B models (plus "Sharp" variants) in FP16, FP8, **GGUF Q8/Q4** and now **INT8 ConvRot** — take a 256px webcam relic to 1024px and beyond with details that GAN upscalers simply cannot invent.

- **Auto Tune for Max Quality:** one button tests your GPU and locks in the best batch size, block swap and tiling — with a huge library of pre-measured configs for 6 GB to 32 GB GPUs.
- BlockSwap, VAE tiling, model offloading, working torch.compile, attention backend selection (FlashAttention / SageAttention / SDPA).
- VAE Encode → Upscale → VAE Decode run as **isolated subprocesses**: zero VRAM/RAM leaks on multi-hour jobs.
- First-frame preview, streaming chunk mode, batch folders (videos + images mixed), full resume.

### 2. FlashVSR+ — the speed demon

![FlashVSR+](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/02_flashvsr_dark.png)

Blazing-fast streaming diffusion upscaler (v1.0 and v1.1), often the best choice for 720p → 4K. Tiny, tiny-long and full pipelines, five VAE options including LightVAE, DiT tile counter that tells you exactly how tiling affects speed, Auto Tune, experimental CFG/noise controls — and it upscales images beautifully too.

### 3. SparkVSR — reference-guided restoration

![SparkVSR](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/03_sparkvsr_dark.png)

The famous reference-frame VSR model, made practical: our exclusive **auto-reference system upscales the first frame of every scene chunk with the model of your choice** (SeedVR2 by default) so SparkVSR always has a perfect anchor. Text prompt guidance with cached encodings, BF16 / FP8-Scaled / **INT8 ConvRot**, stage subprocess isolation (about 1/3 the RAM/VRAM of the public release), Auto Tune.

### 4. NVIDIA RTX Super Resolution — instant results

![RTX Super Resolution](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/04_rtx_superres_dark.png)

Driver-accelerated upscaling up to **8x** with minimal VRAM — perfect when you need speed above all. Quality presets up to HIGHBITRATE_ULTRA with its own auto-tune.

### 5. Image-Based GAN Upscalers — classic, instant, sharp

![GAN Upscalers](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/06_gan_dark.png)

Real-ESRGAN, 4x-UltraSharp (V1/V2), AniScale, NomosUni, DAT-2, HAT-L, SPAN and more — with automatic model metadata via spandrel and the Open Model Database. Works on images, videos (with batch-size acceleration) and entire folders.

### 6. RIFE Frame Interpolation + Video Editing

![RIFE](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/08_rife_dark.png)

2x/4x FPS with every Practical-RIFE model from 4.6 to 4.26, batch folders, PNG/JPG sequence export, plus trim / concatenate / speed-change tools. Also available as **Global RIFE**: automatically interpolate the output of ANY upscaler, chunk-safely.

### 7. Face Restoration

![Face Restoration](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/09_face_dark.png)

GFPGAN-powered restoration as a standalone processor (images, videos, batches) or as an automatic post-step inside every upscaler pipeline, with a global strength slider.

---

## Built for Real Work — Not Just Demos

![Resolution & Scene Split](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/05_resolution_dark.png)

**Movie-scale processing is the core design goal.** Automatic PySceneDetect scene splitting, frame-accurate lossless chunking, per-chunk cleanup, disk-space forecasting, and **full resume** — stop anytime (or survive a power cut) and continue from the last finished chunk. You can even scene-split any video without upscaling at all.

![Output & Comparison](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/07_output_comparison_dark.png)

**A complete output studio:** H.264 / H.265 / AV1 with tune options and film grain synthesis, CRF and preset control, 10-bit output, two-pass encoding, pixel format selection, audio copy/re-encode/remove, FPS override, PNG/JPG sequences, quick presets for YouTube/Archival/Editing/Web. Compare results with native sliders, multi-video and multi-image comparison galleries, or generate **animated slider comparison videos** with custom text.

![Queue](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/10_queue_dark.png)

**Per-GPU queue system:** stack jobs from any tab; each GPU processes its own FIFO queue, so a multi-GPU rig runs multiple upscales simultaneously. Every queued job snapshots its settings.

![Health Check](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/11_health_dark.png)

**Self-diagnosing:** health checks for ffmpeg, CUDA, VS Build Tools, driver version, disk space; a Gradio installation scanner; a bundled-repository scanner; VRAM OOM detection with a prominent alert banner and remediation guidance; structured run metadata and command logging.

![Global Settings](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/12_global_settings_dark.png)

**Universal preset system:** one preset stores every setting of every tab. Auto-loads on startup, propagates across tabs, migrates automatically between versions. Global settings for paths, theme, GPU, execution mode and model cache locations.

![Changelog](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/13_changelog_dark.png)

---

## Dark AND Light — Both First-Class

![Light theme](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/14_seedvr2_light.png)

![Light theme output](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/15_output_comparison_light.png)

---

## Why This Instead of the Original Repos or ComfyUI?

- **One-click installers for Windows, RunPod, Massed Compute and SimplePod** — venv, pre-compiled CUDA-13 wheels (FlashAttention, SageAttention, xformers, Triton, torchao), model downloads: all automatic. The original repos require manual dependency battles; several don't even run on consumer GPUs without the optimizations we wrote.
- **Optimizations you won't find upstream:** SparkVSR at ~1/3 the RAM/VRAM of the public code with our BF16/FP8/INT8 weights; SeedVR2 with phase-isolated subprocesses and unlimited-resolution color fix; FlashVSR+ with a rebuilt stable backend; Auto Tune buttons backed by a downloadable database of real measured VRAM configs from 6 GB to 32 GB GPUs.
- **Pipeline features no node graph gives you for free:** scene-based chunking with resume, per-GPU queues, batch folders of mixed videos+images, collision-safe naming, salvage-on-cancel, automatic audio handling, comparison video generation, telemetry you can audit.
- **A real product, continuously maintained:** 60+ releases since February (see the in-app Version History tab), same-week fixes when users report issues, tutorials, and a Discord community.

---

## Requirements & Installation

**Windows:** Python 3.10–3.12, FFmpeg, CUDA 13, cuDNN 9.17+, C++ Build Tools, Git — follow the requirements tutorial once: https://youtu.be/DrhUHnYfwC0 (works on all GPUs from RTX 2000 up). Then run `Windows_Install_and_Update.bat` and `Windows_Run_SECourses_Upscaler_Pro.bat`. Done.

**Massed Compute (recommended cloud):** register via https://vm.massedcompute.com/signup?linkId=lp_034338&sourceId=secourses&tenantId=massed-compute — coupon `SECourses` for all GPUs. Pick the SECourses creator image and follow `Massed_Compute_Instructions_READ.txt`.

**RunPod:** https://get.runpod.io/955rkuppqv4h • **SimplePod:** https://simplepod.ai/ref?user=secourses — follow `Runpod_SimplePod_Premium_Upscaler_Instructions.txt`.

---

## Get It Now

The full app, all future updates, 60+ releases of continuous development, priority support on Discord, and the entire SECourses script library are available to Patreon supporters:

### ➡️ https://www.patreon.com/posts/150202809 ⬅️

Join the SECourses Discord for help and your special rank: https://discord.com/servers/software-engineering-courses-secourses-772774097734074388 • Star our GitHub: https://github.com/FurkanGozukara/Stable-Diffusion • Reddit: https://www.reddit.com/r/SECourses/ • LinkedIn: https://www.linkedin.com/in/furkangozukara/
