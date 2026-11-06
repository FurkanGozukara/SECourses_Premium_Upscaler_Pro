# Made for SECourses Premium Members : https://www.patreon.com/posts/150202809

## Download Link : https://www.patreon.com/posts/150202809

## 4K NEWER TUTORIAL : https://youtu.be/_WT4C78j5-c

[![Watch the video](https://img.youtube.com/vi/_WT4C78j5-c/maxresdefault.jpg)](https://youtu.be/_WT4C78j5-c)

## 4K HD Tutorial : https://www.youtube.com/watch?v=bPWsg8DREiM

### SECourses Ultimate Video and Image Upscaler Pro - SeedVR2 - FlashVSR+ - Face Restoration - Gan Upscalers - Queue System - Fully Automated Movie Restoration and Upscale Studio

#### It has been long waited to have a studio level video and image upscaler app. Today we have publishing the version 1.0 of SECourses Ultimate Video and Image Upscaler Pro. It is supporting SeedVR2, FlashVSR+, Gan based upscalers, RIFE frame interpolation, full queue system, full batch folder processing, scene / chunked based processing and many more. It is fully working on every cloud and consumer GPUs like RTX 2000, 3000, 4000, 5000 series and H100, H200, B200, RTX PRO 6000. We are installing app with latest Torch and CUDA versions atm all fully automatic with pre-compiled libraries. Even Torch compile is fully and automatically working.

#### Download Link : https://www.patreon.com/posts/150202809

Here are the links sorted in numerical order:

![1](https://github.com/user-attachments/assets/23c996d4-4fb5-4fda-adae-11feed4980d9)
![02](https://github.com/user-attachments/assets/07a0c8de-c9c7-4d08-9fbf-0326b24987c4)
![2](https://github.com/user-attachments/assets/d4264f4a-fd27-40be-9830-6e519aa26e2a)
![3](https://github.com/user-attachments/assets/1b7332a2-fabd-45df-87a1-79be418a6b44)
![4](https://github.com/user-attachments/assets/f0881a38-34ee-47fe-8b52-fae93b90059e)
![5](https://github.com/user-attachments/assets/059c42bc-7331-4202-ad24-aeb1b4de52df)
![6](https://github.com/user-attachments/assets/a2549ec5-e887-4d50-b60e-4270f45f378a)
![7](https://github.com/user-attachments/assets/8ed131ca-941c-4843-aefa-7c08d90a4bdb)
![8](https://github.com/user-attachments/assets/f9b7d639-b8db-42bb-898d-62cb4323fca8)
![9](https://github.com/user-attachments/assets/de690274-ea08-4564-bf13-31270923ea6d)
![10](https://github.com/user-attachments/assets/f8bbfa0a-ff36-4734-9da7-3d45e625f746)
![11](https://github.com/user-attachments/assets/ae196c0a-0421-496c-94e5-262364f4db82)
![12](https://github.com/user-attachments/assets/1cf4af56-f66c-43c8-8f1e-1c607983112e)
![13](https://github.com/user-attachments/assets/3574dde3-dc5d-4307-ba19-40e1ec71e0fd)

## Tier 1: Core Product Features (Most Important)

  - Multi-pipeline app with dedicated tabs for SeedVR2, GAN, RIFE, FlashVSR+, Face Restoration, Resolution/Scene Split,
Output/Comparison, Queue, and Health.
- SeedVR2 upscaling for video, single image, and frame-folder inputs.
- SeedVR2 first-frame preview mode.
- SeedVR2 model selection with metadata-aware defaults and constraints.
- SeedVR2 advanced memory/performance controls: offloading, BlockSwap, VAE tiling, compile options, attention backend
selection, model caching toggles.
- GAN upscaling pipeline for images, videos, and frame-sequence folders.
- GAN model metadata system (Real-ESRGAN + Open Model Database + spandrel-based detection fallback).
- RIFE interpolation pipeline with model selection, multiplier/target FPS modes, spatial scale, precision controls,
static-frame skip, recursion depth.
- FlashVSR+ diffusion upscaling pipeline with version/mode/scale selection and VRAM-aware modes.
- FlashVSR memory controls: VAE/DiT tiling, tile size/overlap, unload DiT before decode.
- RelayVSR (V10.0): fast 4x video and image upscaler. A large Wan model writes a reference keyframe every 15 frames and a small Flash decoder relays its detail to every frame, streaming one frame at a time (the 306-frame 960x540 test clip to 4K in 72 seconds on an RTX 5090, about 4x faster than FlashVSR+ Turbo). Wan INT8 ConvRot (5.1 GB, default, made for the app) or BF16 (10 GB); models download automatically from the SECourses Hugging Face mirror. Auto Tune keeps as many Wan blocks and their keyframe memory on the GPU as fit and streams the rest from RAM, so smaller GPUs give the same result. Batch, preview, chunking with resume, presets, queue and Chains.
- Open-H3 Regenerate (MiniMax H3 + Open-H3-Regenerate-2K adapters): generative 2K/1080p/4K regeneration with a standalone engine, first-click model download, batch and chunked runs in one engine session.
- DLSS 5 Neural Rendering (V9.7): NVIDIA's DLSS 5 renderer for existing videos, images and image folders, with photoreal lighting, skin, hair and material detail, temporal history with optical-flow reprojection and scene-cut resets, style profiles and strength controls. Runs as a PyTorch network on any RTX 30 / 40 / 50 GPU (about 3 GB of VRAM at 4K since V9.8); the model downloads automatically. It can also run automatically after every upscaler as a post-process.
- Chains (V9.7): pipelines of several tabs (for example SeedVR2 2x, then FlashVSR+ 2x, then DLSS 5), each step using its own tab's settings, saved as chain presets and run on one video, several uploads or a folder. Since V9.8: input preview with the size after every step, a quick test on the first seconds, Resume of a stopped run, a viewer for every step's result, and app presets that remember the chain preset. Since V10.0: batch folders (input and output folder, skip finished files, subfolders, a report of every file) and multi-pass DLSS 5 (DLSS 5 two or three times in a row for a stronger look, also as the DLSS 5 tab's Passes setting).
- JonXL H3 Upscale (V10.0): a second Open-H3 workflow, JonXL's MiniMax H3 upscale LoRA (h3_upscale_1920, fixed prompt, 20 steps er_sde / beta57): each picture or video frame upscaled up to 1920 px with real detail from blurry footage; the LoRA downloads on first use.
- Optional launch arguments (V10.0): `--server <IP>` and `--port <number>` (for example `Windows_Run_SECourses_Upscaler_Pro.bat --server 0.0.0.0 --port 7870`); without them the app starts as before.
- Folder and file pickers (V10.0): every path field has a 📂 / 📄 button that opens a native dialog at the current path (hidden on servers without a screen).
- Face Restoration as both standalone processor and integrated post-processing across pipelines.
- Global face restoration enable + global strength control from Global Settings.
- Unified cross-model resolution strategy: Upscale-x, max-edge cap, and pre-downscale-then-upscale behavior.
- Instant Auto Tune for Max Quality on SeedVR2, FlashVSR+, RelayVSR, Open-H3, LTX 2.5, SparkVSR and RTX Super Resolution: sets the largest batch / chunk / window and the least offloading and tiling that fit GPU total - in use - Save VRAM, from measured memory formulas (no test runs); optionally applied automatically on every Upscale.

##  Tier 2: Output, Encoding, and Comparison

  - Output format controls: auto/mp4/png and PNG-sequence toggles.
- Video encoding controls: codec, pixel format, CRF quality, preset speed.
- Audio controls: copy/re-encode/remove plus bitrate.
- Two-pass encoding option.
- FPS override support.
- Global RIFE post-process for all upscaler outputs (multiplier, model, precision, CUDA device).
- Chunk-safe Global RIFE mode (process per chunk before merge).
- Comparison mode selector: native/slider/side-by-side/overlay-style options.
- Image comparison and custom HTML video comparison slider outputs.
- Generated input-vs-output comparison video (auto/horizontal/vertical layout).
- Direct “compare any 2 videos” tool with fullscreen-capable slider UI.
- Pin/unpin reference frame for iterative comparisons across runs.
- Fullscreen and zoom-oriented comparison preferences.
- Metadata/telemetry settings exposed in Output tab.

##   Tier 3: Scale, Throughput, and Long-Run Processing

  - Universal PySceneDetect-based chunking for all major pipelines.
- Static chunking fallback (fixed seconds + overlap).
- Frame-accurate split option (lossless path) vs fast keyframe-based split path.
- Per-chunk cleanup option to save disk.
- SeedVR2 native streaming chunk mode (frame-count chunks) in addition to universal chunking.
- Resume interrupted chunk runs from partial outputs.
- Batch processing in SeedVR2, GAN, RIFE, FlashVSR+, RelayVSR, and Face standalone flow.
- Batch overwrite/skip behavior controls.
- Per-item batch output folder organization and run-dir management.
- Application-level FIFO processing queue across tabs (single active job, waiting jobs).
- Queue monitor tab with refresh, delete selected waiting jobs, and clear-all waiting.
- Queue snapshot isolation so queued jobs run with their captured settings.
- Queue-disabled mode option to ignore extra clicks while busy.

##  Tier 4: Universal preset system that stores all tab settings in one preset file.
- Auto-load last-used universal preset at startup.
- Shared-state sync so loading one preset propagates to all tabs.
- Per-tab model-context presets still supported (save/load/delete/last-used).
- Preset migration/backward compatibility and auto-merge with new defaults.
- Preset guardrails (auto-corrections for invalid combos).
- Global settings persistence (global.json) for output/temp/telemetry/queue/mode/model-cache paths.
- Named global config profiles (save/load/delete) in Global Settings.

##   Tier 5: Reliability, Safety, and Recovery
- Default subprocess execution mode with strong cancellation and cleanup behavior.
- Confirm-cancel safety checkbox in processing tabs.
- Save VRAM (GB) on every upscaler (like ComfyUI --reserve-vram): each run is capped at GPU total - in use - Save VRAM and never spills into shared GPU memory.
- VRAM OOM detection and prominent OOM alert banner with remediation guidance.
- Best-effort salvage of partial outputs on cancellation/failure.
- Collision-safe output naming to avoid overwrite.
- Structured per-run metadata and telemetry logging.
- Command logging for executed processing commands and failures.
- Input/settings validation guardrails (batch-size constraints, tile overlap constraints, GPU spec validation,
compatibility checks).
- Automatic ffmpeg-related checks and fallbacks in multiple paths.
- Audio preservation/replacement utilities for merged/chunked outputs.

##   Tier 6: System Diagnostics and Environment Control

  - Health checks for ffmpeg, CUDA, VS Build Tools, writable dirs, and disk space.
- Gradio compatibility and source scan reporting.
- Repository scan for bundled external components (SeedVR2, Real-ESRGAN, Open Model Database).
- GPU detection utilities designed to avoid unwanted parent-process CUDA context allocation.
- Model discovery registries for SeedVR2/RIFE/FlashVSR/GAN.
- Launcher environment variable integration (TEMP/TMP, MODELS_DIR, HF_HOME, TRANSFORMERS_CACHE, HF_DATASETS_CACHE).
- Editable model cache path controls in Global Settings with restart guidance.

##   Tier 7: UX and Operational Convenience (Least Important)

  - File upload plus manual path entry in all major tabs.
- Auto media preview (image/video) in tabs.
- Input auto-detection (video/image/frame sequence/directory) with diagnostics.
- Missing-frame detection for frame-sequence inputs.
- Resolution calculator and chunk estimator panels with disk-space warnings.
- “Use SeedVR2 input” quick-link from Resolution tab.
- Open output folder and clear temp-folder actions from tabs.
- Live progress indicators, chunk progress text, and ETA/status messaging.
- SeedVR2 model status panel with refresh and auto-refresh.
- Manual CUDA cache clear actions from UI.
- --share launch flag support for Gradio sharing.
