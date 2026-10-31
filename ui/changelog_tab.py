"""
Version History / Changelog tab.

Every release lives in its own accordion (newest first, newest open by
default) so users can quickly scan versions and expand only what they care
about. Written for regular users - plain language, no expert jargon.
"""

import gradio as gr

# Each entry: (accordion title, markdown body). Newest first.
CHANGELOG_ENTRIES = [
    (
        "V9.2 — 1 November 2026 — Fixes: FlashVSR+ Auto Tune, Variable-Frame-Rate Videos, Faster Model Downloads, GAN Architectures",
        """
**A fix release from a full test of every tab: FlashVSR+ Auto Tune no longer plans runs that run out of VRAM, phone and screen recordings upscale with the right timing, big model downloads start at once, and cancelling a job never touches an older run.**

- **Fixed: FlashVSR+ Auto Tune could plan a run that ran out of VRAM part-way** on big outputs close to the GPU's limit (a 1280x1920 portrait video on a 32 GB GPU stopped at 22% and restarted in tiles). It now keeps the extra room the LightVAE decoder really needs (measured with portrait and landscape videos), so the plan fits the first time.
- **FlashVSR+ tiles are up to 2x faster.** When the frame needs DiT tiles, Auto Tune still picks the fewest tiles (fewest seams) but now at the smallest tile size that keeps that count; larger tiles only repeated work in the overlaps. That portrait video: 2 tiles in 75 s instead of 146 s, with the same picture and 5.7 GB less VRAM.
- **Fixed: SparkVSR stopped with a timing error on phone and screen recordings with a variable frame rate.** The app read their nominal rate (for example 30 fps for a clip that really plays at 24), so the upscaled video came out shorter than its sound. Every tab now reads the real average rate of such videos; normal constant-rate videos are unchanged.
- **Model downloader (new in the V9.2 zip): large files start at once.** Before the first byte, an 8 GB model sat for about 40 seconds while Windows filled its empty download file with zeros; that wait is gone. The RIFE download no longer fetches stray macOS `.DS_Store` files.
- **Fixed: the GAN tab left an extra `_upscaled` copy of each image in your input folder** when face restoration was on. Everything now stays in the run's output folder.
- **The GAN tab now names each model's architecture** (DAT, ESRGAN, HAT, RCAN, SPAN, OmniSR and more) instead of "unknown". It reads only the tensor names in the model file, so the tab still opens instantly; models you have not downloaded yet show the architecture they will use.
- **Face Restoration:** choosing CodeFormer silently skipped restoration (that backend is not available yet); it now uses GFPGAN and says so.
- **Fixed: cancelling a job before its first chunk finished saved a "cancelled_partial" video made from an older, finished run** (with the new input's sound) inside that run's folder. Cancel now only saves what the cancelled job itself finished.
- Cancelling a SeedVR2 run now says "Cancelled by user" instead of a long Windows exit code, the Auto Tune result window no longer spills past its border, and the Windows installer no longer prints a warning about the scene detection package.
""",
    ),
    (
        "V9.1 — 30 October 2026 — Instant Auto Tune & Save VRAM for Every Upscaler, Never Shared GPU Memory",
        """
**Auto Tune is now instant on all six AI upscalers, and no upscaler ever spills into shared GPU memory again.**

- **New Auto Tune for Max Quality on SeedVR2, FlashVSR+, Open-H3 Regenerate, LTX 2.5, SparkVSR and RTX Super Resolution.** It no longer runs test clips. One click reads your GPU's total VRAM and what is already in use, your input's size and length, your output size and the selected model, and sets the best settings in under a second:
  - **SeedVR2:** the largest **Batch Size** that fits (the whole clip in one batch when possible; more frames per batch = steadier video, less flicker), split into even batches (72 frames run as 37 + 35, not 65 + a short, flicker-prone 7; measured: batch 121 scored +0.32 dB and 4% less flicker than batch 21 on the same 121 frames, and up to 601 frames are safe because the model's attention positions are per window), then the lowest **BlockSwap**, then VAE tiling only if needed, with the largest tiles.
  - **FlashVSR+:** the longest **Frame Chunk Size** (the whole clip in one pass when possible, up to 4,000 frames per pass: longer passes overran the model's position table and crashed, so longer clips now stream in passes), **DiT tiling off** when the full frame fits (otherwise the largest tiles with a wider overlap), KV Ratio 3 and no CPU offloading.
  - **SparkVSR:** **33-frame Temporal Chunks**, the length that measured best: 720p and 4:3 videos were downscaled 4x, upscaled back and compared with the originals, and 33-frame chunks scored highest on both (+0.15 dB over 65-129 frames, less flicker), while longer chunks were only faster. Shorter chunks also need less VRAM, so 1080p now runs without spatial tiles on 24 GB GPUs. Then **no spatial tiling** (or the largest tiles, now up to 1536 px: 4K output on a 24-32 GB GPU runs 6 tiles instead of 12, fewer tile seams), block offload only when the BF16 model does not fit (with the INT8 ConvRot model block offload uses more VRAM, so Auto Tune never picks it and suggests the BF16 model on cards where INT8 cannot fit), and **Stage Subprocess Isolation off** when the extra 1 GB and the 8.9 GB text encoder fit (one process is about 35% faster; on smaller cards the separate text stage streams the encoder from RAM). Good to know: the INT8 ConvRot SparkVSR model uses *more* VRAM than BF16 on long chunks (18.8 vs 16.0 GB for 65 frames at 1080p), and Auto Tune accounts for it.
  - **LTX 2.5** and **Open-H3:** automatic token budget with the longest chunk/window, so the whole video runs in one pass whenever it fits, with the plan shown in the report.
  - **LTX 2.5 chunks stop at 20 s** (481 frames at 24 fps, 601 at 30): the model's time positions cover 20 s, and a single 24 s chunk scored lower than 20 s chunks across the whole video. Max Chunk Frames above 20 s now logs a note.
  - **LTX 2.5 runs much longer chunks now.** Its VRAM estimate per frame was re-measured and was about 5x too cautious. On a 32 GB GPU, 121 frames of 720p input (2560x1440 output) now run in **one chunk instead of 5**, and 1080p input (4K output) in 3 chunks of 49 frames instead of 15 chunks of 9 frames: fewer chunk seams, steadier video. 360p input with 121 frames finishes about 20% faster because more transformer layers stay on the GPU.
  - **RTX Super Resolution:** the highest quality level of the family you picked (Standard, High Bitrate, Denoise or Deblur) that fits.
- **Save VRAM (GB), default 1.5 GB, on every upscaler tab** — like ComfyUI's `--reserve-vram`. A run uses at most *GPU total − VRAM in use − Save VRAM*. Example: a 31.8 GB GPU with 1.5 GB in use and Save VRAM 1 GB gives the run up to 29.3 GB.
- **New "Auto Tune on every Upscale" checkbox (on by default).** Each Upscale first applies Auto Tune for that exact input (each file of a batch, each scene chunk) and the VRAM that is free at that moment, then starts. Queued jobs are sized when they actually start, not when you clicked.
- **Shared GPU memory is never used.** Every SeedVR2, FlashVSR+ and SparkVSR process (including each phase subprocess) and the LTX 2.5 and Open-H3 engines are capped at that budget, so a request past it stops with a normal out-of-memory error instead of Windows silently moving VRAM into system RAM (about 20x slower). RTX Super Resolution checks the budget before it starts. Verified with Windows' per-process GPU memory counters: shared GPU memory stayed flat in every test, including runs deliberately pushed past the limit.
  - **If a run still reaches its limit** (for example an unusual input needed more than estimated), SeedVR2, SparkVSR and LTX 2.5 re-plan with a smaller budget and run again automatically, and **FlashVSR+** switches to DiT tiling and then smaller tiles instead of failing.
  - Runs that are planned close to their budget use a memory-allocator setting that avoids fragmentation (without it a 1080p SeedVR2 batch failed with 4 GB of VRAM stuck in unusable pieces); runs with room to spare skip it, because it costs about 10-15% speed.
  - **LTX 2.5** streams text-encoder and transformer layers from RAM when they do not fit, and **SparkVSR** streams its text encoder when a new prompt has to be encoded on a small budget.
- **Tested on an RTX 5090 (32 GB) and an RTX 3090 (24 GB)** with 1080p to 8K outputs and every aspect ratio, plus 8, 10, 12 and 16 GB cards simulated with Save VRAM. Smallest GPU each upscaler ran on at the default Save VRAM 1.5 GB: **SeedVR2 and RTX Super Resolution 8 GB; SparkVSR 10 GB with the BF16 model, 12 GB with INT8 ConvRot; FlashVSR+, LTX 2.5 and Open-H3 12 GB.** On a smaller card, lower Save VRAM and close other GPU apps; a run that does not fit stops with a clear out-of-memory message.
- **Model downloader: pick any presets in one go.** `Windows_Download_Models_and_or_Resume.bat` now opens a menu of 26 presets instead of always downloading the regular set: type one number (`5`), several (`1,3,8`) or ranges (`1-3,15-18`) and everything selected downloads in one resumable run, with shared files fetched once. New presets for the app's default models: SeedVR2 7B Sharp INT8 ConvRot (~8.5 GB), SparkVSR INT8 ConvRot (~16 GB), the LTX 2.5 default set (~38 GB) and the Open-H3 default set (~55 GB). `Models_Downloader.py --select "1,3-5"` does the same without asking.
- **Open-H3 Regenerate: faster sparse attention that keeps the picture's structure, and lip sync on talking videos.**
  - **Sparse attention now uses the upstream quality preset** (first refine step dense, the rest sparse at Tau 1.0, from NVIDIA's Sol-Attn update). Measured at 2K: the same structure as full dense attention (37 dB apart) with a 23% faster refine (34 s vs 44 s). The previous setting and the upstream fast preset ran the first, high-noise step sparse, which reshaped objects (a robot arm changed shape). New **Sol-Attn Dense Steps** control: 1 = quality (default), 0 = all sparse (fast, about 30% quicker refine, objects can reshape).
  - **Audio Conditioning now defaults to Source soundtrack.** The refine attends to the video's real audio instead of encoded silence, so talking mouths stay in sync: on a talking-head test every 2-second segment matched the original mouth better (+0.65 dB on the mouth area), and with silence the lips opened where the speaker's were closed. Videos without audio still use silence.
  - A second same-size refine pass for 4K was tested and not adopted: it doubled the refine time and added grain and invented objects.
- **FlashVSR+: the VAE choice now works, and Full mode uses the best decoder by default.**
  - Every VAE was loaded with none of its weights (a file-format mismatch), so Full mode quietly decoded with the tiny modes' TC decoder, and tiled VAE produced the colour noise that a V9.1 guard had been hiding. VAEs now load completely and Full mode decodes with the selected VAE, as upstream FlashVSR does.
  - **New default: LightVAE_W2.1.** Two test videos (720p footage and a talking head) were downscaled 4x, upscaled back and compared with the originals: LightVAE scored best on every measure (+0.3 to 0.4 dB over the TC decoder, +0.6 dB over the full Wan2.1 VAE, least flicker) and was the fastest. The other decoders over-sharpen and invent detail, for example thick black rims on thin metal glasses.
  - Only **Wan2.1** and **LightVAE_W2.1** remain: Wan2.2, TAE_W2.2 and LightTAE_HY1.5 were built for other models' latents and cannot decode FlashVSR's (saved presets that name them use LightVAE_W2.1). The model downloader no longer fetches those files, which saves a 2.8 GB download. In tiny / tiny-long mode the VAE choice is greyed out, because those modes decode with the TC decoder.
  - Auto Tune accounts for the decoder (+0.44 GB per output megapixel for LightVAE) and decodes the full Wan2.1 VAE in tiles, which otherwise needs about 8 GB per output megapixel.
- **Fixed: FlashVSR+ INT8 ConvRot silently ran in BF16.** The FlashVSR engine had no INT8 support, so the Windows default fell back without saying so. INT8 now really runs, from the prebuilt cache the model downloader fetches (1.75 GB instead of 2.84 GB of weights): about 1 GB less VRAM, with the same quality as BF16 (within 0.02 dB on both test videos).
- **Fixed: a FlashVSR+ tiny-mode run that ran out of VRAM finished but wrote no video** ("Streaming decode did not produce any output frames"): the automatic switch to DiT tiling returned the frames in a way the writer ignored. They are now written, without repeating frames written before the switch.
- **Fixed: RTX Super Resolution wrote MPEG-4 Part 2 video through OpenCV.** It now encodes with FFmpeg using the Output tab's codec settings (H.264 by default; H.265, AV1 and VP9 as selected). OpenCV is only a fallback when FFmpeg is missing.
- **Fixed: model downloads failed at once on a brief network or DNS hiccup** (`[Errno 11001] getaddrinfo failed`). The newer Hugging Face library reports network errors differently, so the downloader treated them as permanent and showed the raw error. Looking up each file is now retried with backoff (about 30 s in all), so a short drop recovers by itself. If this PC really cannot look up Hugging Face, the message now says so and what to check: the connection, VPN, proxy, firewall, antivirus web filter or ad-blocking DNS (allow huggingface.co and hf.co, or switch DNS), or `HF_ENDPOINT` where Hugging Face is blocked.
- **Fixed: SeedVR2 used up to 6 GB of shared GPU memory on large batches.** Its runner selected a memory allocator that ignores memory limits; the limit now works.
- **Fixed: Universal Preset "Reset All" (and loading a preset) could leave other tabs showing old values** when you had edited them by hand and the preset's values equalled the ones that tab last showed. Every load and reset now refreshes each tab when you open it.
- **Fixed: Open-H3 at 4K failed on 24 GB GPUs** after the whole video had been regenerated: the fast Sol VAE decoder needs about 21-25 GB for a 4K window. When a window does not fit, it is now decoded in overlapping 1536 px tiles (smaller if needed), blended like ComfyUI's tiled decode: identical to the full decode within 49 dB PSNR, no visible seams, about 14 s per 4K window.
- **Fixed: LTX 2.5 counted its INT8 ConvRot transformer at twice its real VRAM size,** so it kept half of the layers that fit in RAM; now more layers stay on the GPU: a 73-frame 720p chunk takes 25.5 s instead of 38.4 s on an RTX 5090.
- **Fixed: FlashVSR+ reported a run that stopped part-way (for example out of VRAM) as finished** and kept the truncated video. It now reports the error.
- **FlashVSR+ automatic out-of-memory recovery** tiles the DiT first (then smaller tiles) and tiles the decoder last.
- **Fixed: FPS Override crashed FlashVSR+ runs** before they started.
- **Fixed: some FlashVSR+ and SparkVSR batch image runs crashed** while preparing their work folder.
- **Fixed: Face Restoration after RTX Super Resolution never ran.**
- **Fixed: on PCs with two different NVIDIA GPUs, the GPU picked in the app could differ from the one actually used.** GPUs are now numbered the same way everywhere.
- **SparkVSR:** a saved Temporal Chunk Length of 0 (whole clip) no longer shows as 65 when the app starts.
""",
    ),
    (
        "V9.0 — 30 October 2026 — NEW: Open-H3 Regenerate Upscaler & a Faster, Redesigned Interface",
        """
**A new generative upscaler joins the app, and the whole interface is redesigned and much faster.**

- **NEW: 🌀 Open-H3 Regenerate tab** (right next to FlashVSR+) — the MiniMax H3 video model with NVIDIA's Open-H3-Regenerate-2K adapters, running on our own standalone engine (no ComfyUI needed). Your video is encoded at a small draft size, upscaled in latent space onto the target size and regenerated in a few diffusion steps, so it adds real new detail (skin, fabric, foliage, hair) instead of only sharpening.
  - **Target sizes:** 2K (1440p, the trained target), 1080p, 4K, 768p (refine only) or a custom upscale factor, always keeping the aspect ratio.
  - **Models:** FL2VA Pruned INT8 ConvRot (recommended default), FP8 Scaled, W6A8 and BF16; text encoders Qwen3-VL 32B NVFP4 (default), INT8 ConvRot and BF16; the fast Sol VAE decoder. Everything downloads on your first click into the `OpenH3_Models` folder (about 55 GB for the default set). `Models_Downloader.py` has a new Open-H3 option too.
  - **Regenerate strength (Start Sigma), default 0.8:** keeps every subject, object and colour faithful while adding detail. Higher values change more; 0 uses the original release schedule.
  - Steps, seed, adapter strengths, sparse attention, token budget (auto from your free VRAM or manual), window length with seamless cross-faded overlaps, source or 24 fps timing, audio handling and start/end frame are all adjustable. **Preview First Frame** checks the look in about a minute and **Preview Window Plan** shows how the video will be split before you spend GPU time.
  - All the shared features of the other upscalers: images and videos, batch folders (all files run in one engine session, so the models load only once), scene chunking (also one engine session for all chunks), presets, image slider and side-by-side comparison video, queue, GPU selection, audio passthrough, face restoration, Global RIFE, auto transfer to input, resume folders and run metadata.
  - **Never uses shared GPU memory** (on Windows that is about 20x slower). The new **Save VRAM (GB)** option (default 2 GB, like ComfyUI's `--reserve-vram`) keeps that much VRAM free, and the engine never asks for more than the rest. When a step needs more, it automatically moves just enough DiT blocks to system RAM and redoes that window, so the run still completes, with as many blocks as possible kept on the GPU for speed. **Max DiT Blocks on GPU (Block Swap)** lets you lower VRAM use further.
  - **Optional: run each stage as a separate process** (first option under Open-H3 Models). Text encoding, VAE encode + latent upscale, DiT refine and VAE decode each run in their own process and pass their results to the next one through files, so a finished stage keeps no RAM or VRAM.
  - The FFN Chunk Tokens setting now really takes effect, and free VRAM is measured correctly on Windows even when other apps use the same GPU.
- **Redesigned interface for both dark and light themes:** a new header with a one-click theme switch, every tab visible in a tidy grid, clear panels and borders, easy-to-read labels and help text, coloured buttons that look right in both themes, and readable health-check and sizing cards in light mode.
- **Much faster and smoother** (measured in Chrome):
  - Switching tabs paints in 30–120 ms instead of 340–390 ms, and the 0.5–4.4 second freezes after a switch are gone.
  - The page no longer works in the background while idle (before, two timers kept it about 8% busy all the time).
  - The page downloads 60% less data (1.8 MB instead of 4.3 MB) and opens about 25% faster.
  - Preset save and load finish in about 0.1–0.4 seconds instead of 1.2–1.5 seconds, and each setting change sends fewer requests.
  - The app itself starts about 0.7 seconds faster.
""",
    ),
    (
        "V8.6 — 13 October 2026 — RIFE Chunk Merge Fixes",
        """
**Fixed RIFE runs that stopped at the very end with "Merge failed" or "Merged video timeline does not match the source", and RIFE outputs that silently lost frames.**

- **Videos whose audio ends before the picture no longer fail.** RIFE's own audio step cut the interpolated video to the length of the audio, so the last chunk came out short and the final check rejected the whole run at about 96%. RIFE now writes the video only and the app adds the audio itself, without ever shortening the video. This also fixes the RIFE tab and single-pass Global RIFE, which could lose the end of the video the same way.
- **MPEG-TS camcorder and broadcast files (.ts, .m2ts, .mts) now work correctly.** Their frame rate, size and video length were read wrongly, so chunked RIFE dropped interpolated frames at every chunk boundary and chunk splitting needed slow repair steps.
- **MKV and WebM sources with a longer audio track** are now measured by their video length instead of the audio length.
- **A more robust final merge.** If OpenCV cannot read a chunk exactly, the merge automatically retries with FFmpeg's own decoder instead of failing. Duplicate boundary frames are matched chunk by chunk, and merge warnings now name the exact problem (frame count, duration, frame rate or start time) instead of always saying "duration drift".
- **No more freeze while splitting.** A chunk file that ended up with sound but no picture could make the app hang forever while checking it. Such files are now rejected immediately.
- **RIFE tab: a Target FPS above the multiplier now works.** For example 24 fps with x2 and Target FPS 60 stayed at 48 fps. RIFE now interpolates x4 and resamples to exactly 60 fps.
- **The Windows updater now really updates the app.** `Windows_Install_and_Update.bat` updated RIFE, SeedVR2 and the other model folders but never the app itself, so existing installations kept running their old version after "updating". Download the new zip once and run it; from then on updates arrive normally.
- Verified with real RIFE 4.26 runs: 24, 23.976, 25, 29.97 and 60 fps sources; x2, x4 and x8 (up to 480 fps); automatic scenes, a single chunk, 20 chunks and chunk overlap; variable frame rate, MPEG-TS and MKV input; H.265 10-bit and AV1 output; audio shorter or longer than the video; Target FPS above and below the multiplier; and Global RIFE, chunked and single pass. Every result has the exact interpolated frame count, frame rate and length, starts at zero and keeps its audio.
""",
    ),
    (
        "V8.5 — 4 October 2026 — 16-bit Images, Non-English File Names & GAN Video Fixed",
        """
**Fixed upscales that came out almost completely white or stopped with an error: 16-bit images, non-English file names and GAN videos with FFmpeg 9.**

- **SeedVR2: 16-bit images no longer come out white.** Many AI-generated PNGs and 16-bit TIFFs store colors with extra precision. SeedVR2 read them like normal images, so the result was an almost pure white picture. They are now read correctly, including grayscale and transparent ones, in single images, image folders and frame sequences. Grayscale single images, which could stop with an error, also work now.
- **GAN image upscalers: 16-bit grayscale images no longer come out white.** 32-bit float TIFF images, which could not be opened before, now work too.
- **FlashVSR+, SparkVSR and RTX Super Resolution:** 32-bit float TIFF images now work instead of stopping with "Failed to read input image". SparkVSR's automatic reference image is also read correctly from 16-bit grayscale inputs.
- **GAN video upscaling works with FFmpeg 9.** FFmpeg 9 removed an option the GAN tab used when splitting a video into frames, so GAN video upscales stopped with an error. The new option is used now, and older FFmpeg versions keep working.
- **FlashVSR+: videos without a stored length now upscale.** WebM and MKV recordings saved by browsers, or by live and interrupted captures, do not store their duration, so FlashVSR+ read 0 frames and produced no output. FLV videos lost their last frame. FlashVSR+ now reads every video to its real end.
- **File names with Turkish or other non-English letters (ç, ğ, ı, ö, ş, ü, é, 日本...) now work in every model.** Such images could fail to open in SeedVR2, FlashVSR+, SparkVSR and RTX Super Resolution, and resized copies, face restoration results and saved images could be written under a garbled name and then go missing.
- Verified with every image model (SeedVR2, GAN, FlashVSR+, SparkVSR, RTX Super Resolution) on 11 kinds of image files: normal, 16-bit color, 16-bit grayscale, 16-bit with transparency, grayscale, grayscale with transparency, palette, 16-bit TIFF, float TIFF, CMYK JPEG and a Turkish file name. The new `tools/image_input_smoke.py` repeats this check for future versions.
""",
    ),
    (
        "V8.4 — 7 September 2026 — RIFE Merge Timing & Validation",
        """
**More reliable final merging for videos processed with RIFE.**

- **Correct playback duration for interpolated chunks**, including 24 → 48 fps with RIFE 4.26. Merge validation uses the output frame rate, so a five-second video is not incorrectly expected to last ten seconds. Slightly different chunk frame counts are handled without repeated failed re-encoding attempts.
- **Consistent timing during fallback merging** — alternate merge attempts now keep the same frame-based chunk durations as the initial attempt.
- **Fixed playback speed above 240 fps** — high frame rates such as 480 fps, including fractional rates, are retained accurately instead of being capped during merging.
- **Frame and audio safeguards retained** — missing frames still fail validation. Verified with a real video in Google Chrome, including chunked RIFE 4.26 processing, complete playback, and unchanged audio.
""",
    ),
    (
        "V8.0 — 13 August 2026 — NEW: LTX 2.5 Upscaler (22B), Gradio 6.24, Faster Interface",
        """
**A brand-new upscaler model joins the app: LTX 2.5 Pixel Spatial Upscaler 2x.**

- **NEW: 🎞️ LTX 2.5 tab** (right next to FlashVSR+) — Lightricks' 22-billion-parameter LTX 2.5 video model with the official IC-LoRA "Pixel Spatial Upscaler", running on our own standalone engine (no ComfyUI needed). It re-renders your video at exactly **2x resolution** with generative detail, faithful to the source.
  - **7 model variants**: Distilled (8 steps, fast) and Dev (20 steps, max quality) bases, each in **INT8 ConvRot (recommended default)**, **INT4 W4A8 ConvRot** (smallest download), **BF16** (highest fidelity), plus **NVFP4** for the Distilled base — natively accelerated on RTX 50 "Blackwell" GPUs and automatically emulated (same output) on older cards. Identical quantization kernels to ComfyUI.
  - **Switching model variants auto-sets the right steps / CFG / sampler** for that variant — and you can still override every one of them manually, including the step count (the distilled schedule resamples cleanly to any step count).
  - **Single-chunk-first VRAM auto-tune**: this model works best when the whole video is processed as ONE chunk. The default "Auto (Single Chunk Preferred)" mode sizes the token budget from your free VRAM and only splits into overlapped, cross-faded chunks when it truly has to — with a "Preview Chunk Plan" button that shows the exact plan (chunk count, token usage) before you spend GPU time. Manual token budget / frame cap / overlap controls mirror the ComfyUI workflow for full control.
  - **ComfyUI-grade VRAM management**: weights stream from system RAM per-step when the model doesn't fully fit (all variants run even on much smaller GPUs), tiled VAE encode/decode, and the Gemma 4 12B text encoder loads, encodes, and unloads before the transformer ever touches VRAM.
  - **Everything downloads on first use** into the easy-to-find `LTX25_Models` folder — only the files your selected variant needs (~38 GB for the default INT8 pair). The standalone downloader also gains an "ALL LTX 2.5 files" option.
  - All the shared features you expect from the SeedVR2/FlashVSR+/SparkVSR tabs: batch processing, presets, scene chunking, comparisons, queue, GPU selection, audio passthrough, metadata, resume folders.
- **Gradio upgraded to 6.24** with an interface performance pass: smoother and faster tab switching, fewer unnecessary re-renders, snappier video previews and progress updates across every tab.
- Processing progress for every model is mirrored to both the console window and the in-app log, always.
""",
    ),
    (
        "V6.1 — 5 August 2026 — INT8 ConvRot V2: Higher Quality, Shippable Caches",
        """
**A pure quality upgrade for INT8 ConvRot on all three diffusion upscalers.** Existing INT8 caches are rebuilt automatically on first use (one-time).

- **Smarter layer selection (all models)** — a new automatic policy keeps the quality-critical conditioning layers (timestep/text embedders, modulation projections, input/output heads) in full precision, exactly like the official recipes for Wan, Qwen-Image and LTX-2 do. The biggest winner is **FlashVSR+**, which previously quantized its timestep-modulation projection - the layer that steers every block.
- **Better quantization math (all models, no calibration needed)** — full [-128, 127] INT8 range, a closed-form least-squares scale refit after the MSE clip search, corrupted-weight sanitization before rotation, and a built-in **low-rank error-recovery adapter (ARA)** stored inside the same single cache file (~1-2% larger, measurably closer to BF16). A small "rescue budget" automatically keeps the few worst-quantizing layers in full precision.
- **Sharper INT8 runtime** — the fused Triton kernel now does its quantization arithmetic in full FP32 (the old BF16 division added small rounding noise on large values), and supports the new optional per-group weight scales.
- **Optional calibration for maximum quality** — run any upscale once with `SECOURSES_INT8_CALIBRATE=1` (plus `SECOURSES_INT8_HESSIAN=1` for GPTQ) on the BF16/FP16 model and the next INT8 conversion automatically uses activation statistics: energy-weighted scales, GPTQ rounding, Hessian-fitted ARA and bias correction.
- **Shippable caches** — INT8 cache files are now validated by content instead of machine-specific file timestamps, so a cache generated on one PC (for example one we distribute) loads instantly on any other PC. If the cache is missing, it is still generated locally on first use exactly as before.
- **New benchmark tool** — `python tools/int8_quality_benchmark.py --model sparkvsr|seedvr2|flashvsr` measures how close the INT8 weights stay to the BF16/FP16 originals (per-layer and model-wide SQNR) and compares the old V6.0 pipeline against the new one.
""",
    ),
    (
        "V6.0 — 2 August 2026 — INT8 ConvRot Quantization, Modernized Interface, Gradio 6.22",
        """
**A massive quality-of-life and performance release.**

- **NEW: INT8 ConvRot quantization for all three diffusion upscalers** — a new precision option next to BF16 and FP8 Scaled, using the same proven technology recently added to ComfyUI and our Musubi Trainer (group-wise Hadamard rotation + per-row MSE-optimized INT8, fused Triton INT8 matmul).
  - **SparkVSR**: new "SparkVSR-int8-convrot" model choice. The transformer-only INT8 model is generated automatically as one reusable safetensors file from the BF16 weights on first use; the text encoder and VAE stay in their original precision. Measured on RTX 5090 (33 frames, 480p to 1080p): **11% faster than BF16 and 26% faster than FP8 Scaled**, about **4 GB less peak VRAM than BF16**, and **better quality than FP8 Scaled** (PSNR vs BF16: 49.1 dB for INT8 vs 47.8 dB for FP8; SSIM 0.9954 vs 0.9948). Transformer file shrinks from 11 GB to 6 GB.
  - **SeedVR2**: new "INT8 ConvRot (DiT)" checkbox (Performance & Compile section). Works with the FP16 models and creates one reusable DiT safetensors cache on first use - about **half the DiT weight VRAM** and around **15% faster sampling** measured on the 3B model. Not needed for GGUF models (those are already quantized).
  - **FlashVSR+**: new "int8_convrot" choice in the Precision dropdown. It creates one reusable DiT safetensors cache on first use and uses about half the DiT weight VRAM with near-BF16 quality (36 dB PSNR). FlashVSR is attention-bound, so treat this one as a VRAM saver rather than a speedup.
  - Works on any NVIDIA GPU from RTX 2000 "Turing" and newer; older GPUs automatically fall back to a compatible slower path with identical output.
- **Modernized interface on Gradio 6.22** — every button on every tab now has its own distinct color so you can find actions at a glance. Smoother tab switching, cleaner look in both themes.
- **Fixed: silent errors** — clicking Upscale with a missing/invalid input used to do nothing visible (the error went to an invisible status box). Errors now appear instantly on every processing tab, and unexpected internal errors show as pop-up toasts.
- **Fixed: fresh installs had a broken FlashVSR+ tab** — the installer never downloaded the FlashVSR runtime code, only its models. All installers (Windows, RunPod, Massed Compute) now set it up correctly.
- **Light theme fixed and polished** — the top navigation bar and health banner now look correct in light mode (dark mode unchanged and still the default).
- **Theme choice now sticks** — switching dark/light in Global Settings is remembered by your browser immediately, even before you save a preset and even after restarting the app.
- **NEW: this Version History tab** — the full release history of the app, right inside the interface.
- Library upgrade pass for Torch 2.13 + CUDA 13 + Gradio 6.22: restored missing `decord`, `sentencepiece`, `av`, `imageio-ffmpeg` dependencies (their absence broke SparkVSR entirely and broke video saving), fixed T5 text-encoder loading with the newest transformers, and updated internal health checks for Gradio 6.
""",
    ),
    (
        "V5.41 — 15 May 2026 — Standalone Scene Split",
        """
- New feature in the **Resolution & Scene Split** tab, added after a user request:
  you can now **scene-split any video** into separate MP4 files **without running any upscaler**.
  Great for preparing long movies before processing.
- To update: just run `Windows_Install_and_Update.bat`.
""",
    ),
    (
        "V5.4 — 8 May 2026 — SparkVSR Text Prompts + Auto Tune Fixes",
        """
- **SparkVSR can now use text prompts** to guide the upscale. After a deep scan, preset prompts were added — and you can type your own custom prompt too.
- Prompts you already used are **cached and reused**, so repeat runs skip the re-encoding step and start faster.
- **Auto Tune for Max Quality** fixed and significantly improved for SparkVSR (better testing logic, more reliable results).
- Fixed SparkVSR resolution bugs, including an edge-case crash with uncommon video resolutions.
- **Auto Upscale First Frame per Chunk** improved: when enabled, the app now upscales the first frame of *all* chunks (based on your Temporal Chunk Length) before the main process starts.
""",
    ),
    (
        "V5.1 — 5 May 2026 — SparkVSR BF16 + FP8 Scaled + Massive RAM/VRAM Savings",
        """
- The public SparkVSR releases were not optimized for home GPUs, so we generated our own **BF16 weights** (20 GB instead of 40 GB) — less disk space, less RAM, faster loading.
- **NEW: Stage Subprocess Isolation** (enabled by default) — VAE encoding, the upscale itself, and VAE decoding each run in their own isolated process. Result: dramatically lower RAM and VRAM usage with zero leaks.
- **NEW: FP8 Scaled model support** — generated automatically from the BF16 weights on first run and reused afterwards. Ultra-fast loading, very low RAM use, lower VRAM use, and no meaningful quality loss in our tests.
- Default Temporal Chunk Length set to 65 for lower VRAM use — and Auto Tune can find the maximum your GPU can handle.
- With all optimizations combined, SparkVSR now uses **about 1/3 of the RAM and VRAM** it used before.
""",
    ),
    (
        "V5.0 — 4 May 2026 — SparkVSR Model Added",
        """
- The famous **SparkVSR** video upscaler joined the app as a full tab with all the shared features (chunking, batch, presets, queue...).
- SparkVSR needs good reference frames to shine, so we built a special system: **the first frame of every chunk is automatically upscaled with the model of your choice** (SeedVR2 by default) and used as the reference.
- Works hand-in-hand with the default scene-based chunked processing.
- **Auto Tune** support included from day one.
""",
    ),
    (
        "V4.8 — 28 April 2026 — FlashVSR+ Batch Fix",
        """
- Fixed: batch-upscaling a folder of **images** with FlashVSR+ was creating unnecessary sub-folders and video files.
""",
    ),
    (
        "V4.7 — 5 April 2026 — Unified Input Statistics + App Icon",
        """
- The input video statistics panel is now identical across all upscalers and also shows the **Output FPS**.
- If the output FPS is changed by FPS Override or RIFE interpolation, the new value is displayed.
- Added a custom **browser tab icon** so the app is easy to spot among your tabs.
""",
    ),
    (
        "V4.6 — 30 March 2026 — Unlimited Color Fix + Per-GPU Queues",
        """
- The **color fix** step after upscaling was completely rebuilt: the old 32-bit size limit is gone, so very high resolution + high batch-size SeedVR2 runs now work. It also uses much less VRAM with identical quality.
- The **queue system is now per-GPU**: with multiple GPUs you can run several upscales at the same time, each GPU with its own queue.
""",
    ),
    (
        "V4.4 — 25 March 2026 — ETA, Cleaner Batches, SeedVR2 Subprocess Phases",
        """
- **ETA estimation** for all upscalers — the app predicts when your whole job will finish.
- **Keep Only Output Files** option for batch runs: temporary chunks and metadata are removed automatically so only final results remain (FlashVSR+, GAN, RTX, SeedVR2).
- **SeedVR2's biggest upgrade yet**: VAE Encode, Upscale and VAE Decode now run as **separate subprocesses** — zero VRAM/RAM leaks and maximum performance. Torch compile is now fully working too.
- RTX Super Resolution: option to disable automatic scene splitting, plus clearer sizing analysis and better progress/cancel behavior.
- RIFE frame interpolation now supports **batch folders** and can export directly as **PNG/JPG image sequences**.
- Faster app startup thanks to lazy-loading of heavy components.
- Many fixes for frame-sequence (image folder) processing.
""",
    ),
    (
        "V4.0 — 14 March 2026 — NVIDIA RTX Super Resolution Added",
        """
- Added the freshly published **NVIDIA RTX Super Resolution** upscaler as a new tab.
- It is **extremely fast**, uses minimal VRAM, and can upscale up to **8x**.
- Its effect is lighter than SeedVR2 or FlashVSR+ — perfect when you want speed. We recommend the `HIGHBITRATE_ULTRA` preset.
""",
    ),
    (
        "V3.6 — 7 March 2026 — Pre-Calculated VRAM Database",
        """
- We actually measured VRAM usage for a huge range of resolutions and aspect ratios on GPUs **from 6 GB to 32 GB**, and the app now downloads these measurements automatically.
- Result: Auto Tune can instantly pick the best-quality settings that safely fit your GPU.
- More fixes for the automatic VRAM optimization buttons for FlashVSR+ and SeedVR2.
""",
    ),
    (
        "V3.3 — 28 February 2026 — Auto Tune for Max Quality",
        """
- **NEW: "Auto Tune for Max Quality - VRAM Optimized"** button for both SeedVR2 and FlashVSR+.
  - Tests parameters against your actual input video and target resolution, then locks in the best quality settings while keeping a safe 2 GB VRAM margin.
  - Results are cached, so future runs with similar output sizes configure themselves instantly.
  - We shipped a large library of pre-made test logs, so it may set perfect parameters for your GPU immediately.
- Final chunk merge now survives inconsistent FPS or odd input formats.
- Fixed a massive memory spike (up to 100 GB) when color-fixing very long single-pass SeedVR2 runs.
""",
    ),
    (
        "V3.0 — 28 February 2026 — Smarter Optimizer + Gradio 6.8",
        """
- The **Optimize Parameters** button became much smarter: upload a video, choose 2x/4x and a target resolution, click — it configures the app for your GPU.
- **Gradio upgraded to 6.8** with noticeable interface smoothness improvements.
- Direct Video Compare: FPS accuracy fixed and a new **text alignment** option added.
- SeedVR2 tuning for low-resolution sources: default model set to the 7B "sharp" variant and default noise lowered to 0 (SeedVR2 shines at very low input resolutions, e.g. 256px → 1024px).
- FlashVSR+ default Frame Chunk Size raised from 64 to 241, plus output naming fix.
""",
    ),
    (
        "V2.91 — 26 February 2026 — DiT Tile Counter",
        """
- FlashVSR+ now shows **how many DiT tiles** will be processed based on your Tile Size and Tile Overlap.
  - Fewer tiles = faster processing and better quality. Maximize Tile Size until VRAM runs short, and increase Tile Overlap if you ever see seam artifacts.
- **Copy output to input** buttons added to SeedVR2, FlashVSR+ and GAN tabs, plus an "auto transfer after upscale" checkbox.
- FlashVSR+ batch runs now display live status in the interface.
""",
    ),
    (
        "V2.8 — 24 February 2026 — Diffusion Controls + New Tutorial",
        """
- New tutorial video published: https://youtu.be/_WT4C78j5-c
- FlashVSR+ gained experimental **Diffusion Controls** (CFG and Noise) for squeezing out extra detail.
- Faster animated slider comparison video generation and much better FlashVSR+ progress logging (single-line progress with speed).
""",
    ),
    (
        "V2.6 — 23 February 2026 — Direct Video Compare",
        """
- New comparison tools in **Output & Comparison → Direct Video Compare**:
  compare **any two videos** with side-by-side layouts or an **animated slider comparison video**, complete with custom text labels.
""",
    ),
    (
        "V2.5 — 22 February 2026 — Statistics Fixes",
        """
- Fixed inaccurate Chunk Stats in some cases.
- Fixed wrong sizing statistics when changing upscale ratio or min/max resolution.
- Improved the information display during processing, and the comparison slider's close button now properly clears uploaded videos.
""",
    ),
    (
        "V2.4 — 20 February 2026 — Global Settings Tab",
        """
- New **Global Settings** tab: output/temp folders, theme, telemetry, queue, GPU selection and more in one place.
""",
    ),
    (
        "V2.3 — 20 February 2026 — All-New FlashVSR+ Backend",
        """
- **FlashVSR+ backend completely replaced** with a heavily upgraded engine — we think it now beats SeedVR2 for high-resolution sources (e.g. 720p → 4K).
- FlashVSR+ can now upscale **images** too, and a low-VRAM button was added for OOM situations.
- New **multi-video and multi-image comparison sliders** in Output & Comparison.
- **H.265 output** with Tune options (grain, psnr, ssim...) added.
- SeedVR2: Max Resolution setting fixed; Latent Noise Scale tip — raise it (e.g. 0.3) for extra new details.
- GAN upscalers now handle videos perfectly with working batch size acceleration.
- Improved top navigation bar and lots of bug fixes across the app.
""",
    ),
    (
        "V1.9 — 18 February 2026 — GGUF Models + Encoding Fixes",
        """
- **SeedVR2 GGUF (Q8) model support** added, with a separate downloader (`Windows_Download_GGUF_and_FP8_Models.bat`).
- Video Output settings (codec etc.) were not being saved with presets or applied to the output — both fixed (this one took a full day of debugging!).
- Fixed frozen frames in GAN-upscaled videos, non-working batch size and chunk previews for GAN.
- FlashVSR+ polished further: live status and progress in both the console and the interface.
""",
    ),
    (
        "V1.8 — 16 February 2026 — GPU Selector",
        """
- **Choose which GPU to use** straight from the top menu.
- FlashVSR+ tab remade, supporting both v1.0 and v1.1 models.
- Health Check now verifies your NVIDIA driver version.
- Every setting is now saved and loaded with your global preset (hidden settings removed).
""",
    ),
    (
        "V1.7 — 14 February 2026 — Unified Preset System",
        """
- Big UI/UX improvement pass.
- **Universal preset system**: every tab and every setting is saved and loaded together with one global preset.
""",
    ),
    (
        "V1.6 — 12 February 2026 — Resume Interrupted Runs",
        """
- **Folder resume**: point the output field at a previous run folder (e.g. `outputs/0017`) and click Upscale — the app continues from the last processed chunk.
- Even if you close the browser mid-run, processing continues and files keep saving.
- Improved the log synchronization between the console and the interface.
""",
    ),
    (
        "V1.5 — 12 February 2026 — GAN Tab Revamp",
        """
- The GAN upscaler page was completely revamped — 10x better looks.
- GAN upscalers (ultra fast!) now work for **both images and videos**.
""",
    ),
    (
        "V1.2 — 10 February 2026 — Batch Folder Fixes",
        """
- Fixed batch folder processing for SeedVR2.
- You can now batch-upscale folders containing a **mix of videos and images**.
""",
    ),
    (
        "V1.0 — 8 February 2026 — Initial Release",
        """
**The first public release of SECourses Ultimate Video and Image Upscaler Pro.**

- SeedVR2 and FlashVSR+ video/image upscaling, GAN upscalers, RIFE frame interpolation.
- Scene-based chunked processing for movies of any length, full batch folder processing, queue system.
- Fully automated installers with model downloads for **Windows, RunPod, SimplePod, Massed Compute and Linux**.
- Works on consumer GPUs (RTX 2000/3000/4000/5000) and cloud GPUs (H100, H200, B200, RTX PRO 6000).
""",
    ),
]


def changelog_tab():
    """Create the Version History / Changelog tab content."""
    with gr.Column():
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.Markdown("## 📜 Version History / Changelog")
            with gr.Column(scale=1):
                toggle_sections_btn = gr.Button(
                    "Open All Sections",
                    variant="secondary",
                    size="lg",
                    elem_classes=["action-btn", "sec-btn-bronze"],
                )
                sections_state = gr.State(value="closed")
        gr.Markdown(
            """
Complete release history of **SECourses Ultimate Video and Image Upscaler Pro** — newest release first.
Click any version below to expand its full changelog.

**Latest zip file, updates and support:** [https://www.patreon.com/posts/150202809](https://www.patreon.com/posts/150202809)
"""
        )

        accordions = []
        for index, (title, body) in enumerate(CHANGELOG_ENTRIES):
            with gr.Accordion(title, open=(index == 0)) as section_accordion:
                gr.Markdown(body)
            accordions.append(section_accordion)

        def toggle_all_sections(current_state):
            if current_state == "closed":
                new_state = "open"
                new_text = "Close All Sections"
                accordion_states = [gr.Accordion(open=True) for _ in accordions]
            else:
                new_state = "closed"
                new_text = "Open All Sections"
                accordion_states = [gr.Accordion(open=False) for _ in accordions]
            return [new_state, gr.Button(value=new_text)] + accordion_states

        toggle_sections_btn.click(
            toggle_all_sections,
            inputs=[sections_state],
            outputs=[sections_state, toggle_sections_btn] + accordions,
            queue=False,
            show_progress="hidden",
        )
