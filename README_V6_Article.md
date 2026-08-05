# SECourses Ultimate Video and Image Upscaler Pro V6.0

## SeedVR2 • FlashVSR+ • SparkVSR • NVIDIA RTX Super Resolution • GAN Upscalers • RIFE • Face Restoration • NEW INT8 ConvRot — The Fully Automated Movie Restoration and Upscale Studio

**Download (Patreon):** https://www.patreon.com/posts/150202809

**4K Tutorial:** https://youtu.be/_WT4C78j5-c • **Quick 4K demo:** https://youtu.be/bPWsg8DREiM

---

Turn any low-resolution video or image into crisp 4K/8K — on your own GPU, with one click. Upscaler Pro is the only app that puts **all seven state-of-the-art upscaling engines** behind one polished interface, installs itself fully automatically on **Windows, RunPod, Massed Compute and SimplePod**, and is optimized so hard that a **120-minute movie upscales end-to-end on a consumer GPU** — with scene detection, resume, queueing, audio handling and color fixing all done for you.

It runs on everything from an **RTX 2000-series** card all the way to **H100 / H200 / B200 / RTX PRO 6000**, with pre-compiled Torch 2.13 + CUDA 13 wheels, working torch.compile, FlashAttention, SageAttention and Triton — zero manual setup. Every screenshot below is a real 4K capture of the app running on an RTX 5090; the whole machine is exposed to the app through a **global hardware selector** that lists every detected GPU plus CPU mode, so a multi-GPU rig can point every tab at exactly the card you want.

---

## NEW in V6.0: INT8 ConvRot Quantization — Faster Than BF16, Better Than FP8

V6 brings the same cutting-edge quantization that recently landed in ComfyUI and our Musubi Trainer to video upscaling: **group-wise Hadamard rotation + per-row MSE-optimized INT8 weights**, executed with fused Triton INT8 tensor-core matmuls.

Measured head-to-head on an RTX 5090 (SparkVSR, 480×272 → 1920×1088, identical settings, re-verified with longer 145-frame multi-chunk runs):

![Benchmark table](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/17_benchmark_table.png)

The chart below shows the same result visually — INT8 ConvRot is the only quantization that is simultaneously **faster than BF16** and **closer to BF16's output than FP8 is**:

![Benchmark chart](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/18_benchmark_chart.png)

- **SparkVSR:** new `SparkVSR-int8-convrot` model choice — **faster than BF16, ~26% faster than FP8 Scaled**, ~4 GB less peak VRAM than BF16, and **closer to BF16 quality than FP8 is** (49.1 dB vs 47.8 dB PSNR). The INT8 model is generated automatically from the BF16 weights the first time you pick it.
- **SeedVR2:** new **INT8 ConvRot (DiT)** checkbox — about **half the DiT weight VRAM** and ~15% faster sampling on the FP16 models, quantized on load.
- **FlashVSR+:** new `int8_convrot` precision — half the DiT weight VRAM with near-BF16 output.
- Works on **RTX 2000 (Turing) and newer**; older GPUs automatically fall back to a compatible path with identical output. No extra downloads, no extra setup.

---

## Also New in V6.0

- **Modernized interface on Gradio 6.22** — every button on every tab has its own distinct color, so you always know where you are and what you are clicking. Tab switching stays fluid even while a heavy upscale is running.
- **Version History tab** — the complete changelog of the app, inside the app, each release in its own expandable section.
- **Polished light theme** (dark stays the default) and **theme choice that sticks** across restarts.
- **No more silent failures** — every invalid input now shows an instant, clear error message on every tab (re-verified this session: empty-input clicks on SeedVR2, FlashVSR+ and GAN all answer within a second), and unexpected errors pop up as toasts.
- **Deep compatibility pass** on the newest Torch 2.13 + CUDA 13 + Gradio 6.22 stack: fresh-install FlashVSR+ setup fixed, RIFE fixed for the current model layout and NumPy 2, GGUF support restored, video saving fixed, SparkVSR dependencies restored.

---

## See the Actual Difference — Real Before / After Media

Numbers are nice; pixels are better. These are real outputs from this machine — a 145-frame low-resolution clip pushed through the engines, shown against the input at matching size (input enlarged with plain nearest-neighbor so you see exactly what the model adds):

![SparkVSR 4x before/after](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/22_cmp_sparkvsr_4x_still.png)

SparkVSR at 4x — 480×272 in, 1920×1088 out. Watch the side-by-side comparison video to see it in motion: [SparkVSR side-by-side comparison MP4](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/22_cmp_sparkvsr_4x_sidebyside.mp4)

![SeedVR2 before/after](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/23_cmp_seedvr2_still.png)

SeedVR2's one-step diffusion invents plausible detail that classic upscalers cannot — the difference is most dramatic on soft, compressed sources. Animated version: [SeedVR2 side-by-side comparison MP4](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/23_cmp_seedvr2_sidebyside.mp4)

![FlashVSR before/after](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/24_cmp_flashvsr_4x_still.png)

FlashVSR+ at 4x with the new INT8 ConvRot precision — the speed-per-quality champion for streaming workloads. Animated version: [FlashVSR side-by-side comparison MP4](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/24_cmp_flashvsr_4x_sidebyside.mp4)

![GAN crop comparison](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/25_cmp_gan_crop.png)

And a classic: 4x-UltraSharp GAN taking a full 1920×1400 frame to **7680×5600** — this is a 1:1 crop from the center of the result.

---

## The Seven Engines — Every Feature, Tab by Tab

### 1. SeedVR2 — the quality king for low-resolution sources

![SeedVR2 tab part 1](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/01a_seedvr2_dark.png)

The legendary one-step diffusion restorer, fully productized. Feed it an uploaded video/image or a manual video, image or **frame-folder path** — the input panel auto-detects the media type and shows previews with resolution / FPS / frame statistics, validation and missing-frame diagnostics. Eight installed models cover every VRAM budget: **3B and 7B, plus "Sharp" variants, in FP16, FP8-mixed and GGUF Q8** — and V6 adds the **INT8 ConvRot (DiT)** checkbox on top. Processing controls include the required 4n+1 batch size, uniform batching, skip/load/prepend frame controls, temporal overlap, seed, input/latent noise (with a still-image custom-noise safeguard) and **six color-correction modes** (LAB, wavelet, adaptive wavelet, HSV, AdaIN, or none). Memory controls go deeper than any other SeedVR2 frontend: separate DiT / VAE / tensor offload devices, BlockSwap with I/O swapping, tiled VAE encode **and** decode with independent tile sizes and overlaps, tile debugging, and a model-directory override. Attention and compilation: SDPA, Flash Attention 2/3, SageAttention 2/3, DiT/VAE torch.compile with Inductor/CUDAGraphs backends, compile modes, fullgraph, dynamic shapes, Dynamo cache and recompile limits — plus DiT/VAE caching and **isolated encode → upscale → decode subprocesses** so multi-hour jobs cannot leak VRAM or RAM.

![SeedVR2 tab part 2](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/01b_seedvr2_dark.png)

The bottom half is where the workflow lives: **Auto Tune with reserved VRAM** picks the best batch size, block swap and tiling for your GPU from a library of real measured configs (6 GB to 32 GB); arbitrary scaling with max-edge limits and optional pre-downscaling; first-frame preview before you commit; full upscale with confirmed cancellation; **scene/chunk resume** and native streaming chunks with partial-chunk resume checks and automatic chunk estimates. Outputs land in a batch gallery with **before/after image and video sliders**, a completed-chunk gallery with click-to-preview, output overrides with auto/MP4/PNG selection, an optional face-restoration postpass, model/CUDA status auto-refresh and one-click CUDA cache clearing (current device or all).

### 2. FlashVSR+ — the speed demon

![FlashVSR+ tab](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/02_flashvsr_dark.png)

Blazing-fast streaming diffusion upscaling, often the best choice for 720p → 4K. Versions **1.0 and 1.1**; **tiny, tiny-long or full** pipelines; five VAE options (**Wan2.1, Wan2.2, LightVAE W2.1, TAE W2.2, LightTAE HY1.5**); auto/BF16/FP16/**INT8 ConvRot** precision; and four attention backends (sparse Sage, block-sparse, Flash Attention 2, SDPA) with seed, frame-chunk, local-range, sparse-ratio and KV-ratio controls. Memory features include CPU model storage, forced offload, VAE/DiT tiling with a **live tile counter** that tells you exactly how tiling affects the run, unload-before-decode and tiny-pipeline stream decoding. Experimental CFG and denoise controls, wavelet color correction, start/end frame trimming, 2x/4x scaling with the shared Resolution settings, Auto Tune with reserved VRAM, batch folders, chunk previews, comparisons, logs — and it upscales images beautifully too.

### 3. SparkVSR — reference-guided restoration

![SparkVSR tab part 1](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/03a_sparkvsr_dark.png)

The famous reference-frame VSR model, made practical. **BF16, FP8-scaled and the new INT8-ConvRot** model variants (with model and LoRA path overrides and bilinear/bicubic/nearest pre-interpolation), noise step / SR noise step / seed controls, and **prompt presets** for faithful restoration, noisy/compressed sources, old film, faces, anime, text/signs — or write your own prompt; text encodings are cached by prompt hash. **Auto Tune** probes your GPU and applies the highest-quality spatial-tile / temporal-chunk settings that fit your free VRAM, with a reserved-VRAM slider and **stage subprocess isolation** (VAE encode, transformer, VAE decode run in separate processes — about 1/3 the RAM/VRAM of the public release). Temporal chunk length and overlap, CPU offload, VAE slicing/tiling, group offload with blocks-per-group, forced offload and debug logging round out the runtime; spatial tiling has live tile-count feedback.

![SparkVSR tab part 2](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/03b_sparkvsr_dark.png)

The reference system is the exclusive part: reference modes **sr_image, pisasr, no_ref and gt** with guidance scale and reference indices; our **auto-reference system upscales the first frame of every temporal chunk** with the upscaler of your choice (SeedVR2 by default, FlashVSR+ or any installed GAN model also selectable) so SparkVSR always has a perfect anchor; a local PiSA-SR backend is fully configurable (Python executable, script, SD model, checkpoint, GPU). Output I/O covers PNG frame saving, YUV444/YUV420 pixel formats, start/end frame ranges, integer scaling up to 16x, shared Resolution settings, output overrides, metadata, a face postpass, batch and chunk galleries, comparisons and logs.

### 4. NVIDIA RTX Super Resolution — instant results

![RTX Super Resolution tab](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/04_rtx_superres_dark.png)

Driver-accelerated upscaling up to **8x** with minimal VRAM — perfect when you need speed above all. **Seventeen quality presets**: standard ULTRA/HIGH/MEDIUM/LOW/BICUBIC plus DENOISE, DEBLUR and HIGHBITRATE variants at four levels each. Its own Auto Tune, arbitrary scaling with shared or local factor, max-edge and pre-downscale controls, nonblocking inference, scene-split override, CUDA stream tuning, streaming chunks and partial-run resume — plus previews, confirmed cancellation, face postpass, batch processing, output overrides, comparison viewer, chunk gallery and logs.

### 5. Image-Based GAN Upscalers — classic, instant, sharp

![GAN tab](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/06_gan_dark.png)

Thirteen installed models — **AnimeSharpV4 Fast, LiveActionV1 SPAN, NomosUni SPAN, AniScale2 Omni, AnimeSharp, UltraSharp, UltraSharpV2, Nomos2 HQ, RealWebPhoto v4, HAT-L, Kim2091 UltraSharp, RealESRGAN x4plus and RealESRGAN x4plus anime** — and the tab dynamically scans your local `.pth`/`.safetensors` folder, reading each model's metadata and native scale through the Open Model Database with a spandrel fallback. Processes images, videos and frame sequences with tile size/overlap, frames-per-iteration, denoising, sharpening, color correction, GPU acceleration, face restoration and per-model output subfolders; arbitrary scaling with max-edge/pre-downscale, batch folders, comparisons, chunk/batch galleries and full resume.

### 6. RIFE Frame Interpolation + Video Editing

![RIFE tab](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/08_rife_dark.png)

The complete motion toolkit, organized in four sub-sections:

![RIFE interpolation](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/20_rife_interpolation_dark.png)

**Frame Interpolation** — nine RIFE models installed (4.14, 4.15, 4.17, 4.18, 4.20, 4.21, 4.22, 4.25, 4.26), x1/x2/x4/x8 multipliers **or an exact target FPS**, spatial scale, UHD mode, FP16/FP32 precision, montage mode, static-frame skipping and recursion depth control — single videos or whole batch folders.

![RIFE editing](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/20_rife_editing_dark.png)

**Video Editing** — trim with start/end times, concatenate with an additional-video list, and speed change, right inside the app.

![RIFE frame control](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/20_rife_frame_control_dark.png)

**Frame Control** — skip the first N frames or cap the number of loaded frames for fast experiments on long sources.

![RIFE output settings](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/20_rife_output_dark.png)

**Output Settings** — MP4/AVI/MOV/WebM containers, H.264/H.265/VP9 with CRF, PNG/JPG sequence export with JPEG quality, audio removal, verbose FFmpeg logging, and a model-directory override. Everything supports process/cancel/resume, comparison sliders and presets — and RIFE is also available as **Global RIFE**, automatically interpolating the output of ANY upscaler tab, chunk-safely (more below).

### 7. Face Restoration

![Face Restoration tab](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/09_face_dark.png)

GFPGAN-powered face restoration as a **global postpass inside every upscaler pipeline** (one enable + strength slider affects all engines) and as a standalone processor with its own six-section interface:

![Face standalone](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/21_face_standalone_dark.png)

**Standalone Processing** — restore single images, whole videos, or recursive batch folders, with restored-image/video downloads, batch downloads and processing logs.

![Face models](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/21_face_models_dark.png)

**Model Selection** — GFPGAN, RestoreFormer, CodeFormer or automatic backend choice.

![Face detection](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/21_face_detection_dark.png)

**Face Detection** — RetinaFace, YuNet, OpenCV or dlib detectors with confidence threshold, minimum face size and maximum face count.

![Face restoration controls](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/21_face_restoration_dark.png)

**Restoration Controls** — strength, blind restoration, face pre-upscale, padding, landmark handling and color correction.

![Face advanced](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/21_face_advanced_dark.png)

**Advanced Settings** — GPU selection, batched face processing and artifact reduction.

![Face quality](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/21_face_quality_dark.png)

**Quality & Output** — output quality, preserve-original mode and saved face masks.

---

## Built for Real Work — Movie-Scale Processing

![Resolution & Scene Split](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/05_resolution_dark.png)

**Movie-scale processing is the core design goal.** Automatic PySceneDetect scene detection with sensitivity and minimum-scene controls (or fixed-duration chunks with overlap when detection is off), frame-accurate lossless re-encoding or fast keyframe splitting, optional chunk cleanup, chunk/scene/duration/disk estimates with disk-space warnings, and standalone numbered-MP4 scene export — you can scene-split any video without upscaling at all. The input picker pulls directly from SeedVR2, GAN or FlashVSR, and the settings propagate to every compatible upscaler. Combined with per-chunk resume, you can stop anytime (or survive a power cut) and continue from the last finished chunk.

## The Output & Comparison Studio — Ten Sections

![Output & Comparison overview](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/07_output_comparison_dark.png)

One tab controls the output pipeline of every engine — here is each of its ten sections:

![Global RIFE](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/19_output_global_rife_dark.png)

**Global RIFE** — post-upscale frame interpolation at x2/x4/x8 for the output of ANY tab, with nine RIFE models, FP16/FP32 precision, original preservation and **chunk-safe interpolation** (chunks are interpolated before merge so seams never appear).

![Output format](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/19_output_format_dark.png)

**Output Format** — auto/MP4/PNG selection, batch overwrite behavior, and **one-click codec presets for YouTube, Archival, Editing and Web**, plus global skip-first/load-cap frame controls.

![Image output](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/19_output_image_dark.png)

**Image Output** — PNG/JPG/WebP with quality control, PNG sequences with filename padding and basename retention.

![Video output](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/19_output_video_dark.png)

**Video Output** — the full encoding studio: **H.264 / H.265 / ProRes / VP9 / AV1** with dynamic pixel formats, optional 10-bit FFmpeg, nine encode speeds, H.265 tunes, **AV1 film-grain synthesis and denoise**, two-pass encoding, CRF control, and audio copy/AAC/Opus/FLAC/removal with bitrate selection.

![Frame processing](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/19_output_frames_dark.png)

**Frame Processing** — FPS override and OpenCV/FFmpeg output engine selection for SeedVR2.

![Comparison display](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/19_output_comparison_display_dark.png)

**Comparison Display** — native, slider, side-by-side or overlay comparison modes with automatic comparison-video creation and automatic/horizontal/vertical layout.

![Direct video compare](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/19_output_direct_compare_dark.png)

**Direct Video Compare** — drop in ANY two videos (upload or path), add custom labels, choose left-right or top-bottom layout, tune live slider height/position, use automatic or custom dimensions with font/alignment control, preview frames, then generate a **comparison MP4 or an animated slider MP4** with cycle duration and slow-motion options.

![Videos comparison pool](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/19_output_video_pool_dark.png)

**Videos Comparison Slider** — a reusable multi-video pool with thumbnail pickers: select any two results as left/right, swap, render, clear, and scrub an interactive HTML slider.

![Images comparison pool](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/19_output_image_pool_dark.png)

**Images Comparison Slider** — the same pool workflow for images.

![Metadata and logging](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/19_output_metadata_dark.png)

**Metadata & Logging** — run metadata saving, telemetry, metadata-format selection and log verbosity — every run auditable, every command recorded.

## Queue, Health, Global Settings & Changelog

![Queue](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/10_queue_dark.png)

**Per-GPU queue system:** resource-aware FIFO lanes per GPU/CPU — stack jobs from any tab and a multi-GPU rig runs multiple upscales simultaneously. Every queued job snapshots its settings; active/waiting tables show lane position and wait time with 2-second refresh and a live tab badge; delete selected waiting jobs or clear all.

![Health Check](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/11_health_dark.png)

**Self-diagnosing:** checks for Gradio, FFmpeg, CUDA/GPU/VRAM, NVIDIA driver 580+, Windows Build Tools, temp/output write access and disk space — with troubleshooting guidance, a Gradio installation scanner and a bundled-repository scanner (SeedVR2 / Real-ESRGAN / OMDB). A persistent health banner and CUDA OOM detection with dismissible guidance keep problems visible the moment they happen.

![Global Settings](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/12_global_settings_dark.png)

**Global Settings & Universal Presets:** default output/temp directories, persistent dark/light theme — and the universal preset system: one preset stores **every setting of every tab**, with select, auto-load on startup, name/save/overwrite, load, reset, delete, last-used restore, automatic schema migration between versions and live cross-tab synchronization (re-verified this session: a full save lands in ~2.7 s across 60+ components).

![Changelog part 1](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/13a_changelog_dark.png)

**Version History inside the app** — 26 dated releases from V1.0 through V6.0, lazy-rendered in expandable sections with an open/close-all control (verified: one click takes 11 open sections to 36).

![Changelog part 2](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/13b_changelog_dark.png)

The release cadence tells the story better than any promise: 60+ releases since February, with new engines, new quantization modes and same-week fixes landing continuously.

## Dark AND Light — Both First-Class

![Light theme SeedVR2](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/14a_seedvr2_light.png)

The complete interface is polished for both themes — and your choice persists across restarts.

![Light theme output](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/15_output_comparison_light.png)

Every tab, every control, every button color works in light mode too.

![Light theme changelog](https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/Upscaler_Pro_V6_Screenshots/16a_changelog_light.png)

Even the changelog accordions render beautifully in light mode.

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
