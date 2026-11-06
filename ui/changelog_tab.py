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
        "V10.0 — 6 November 2026 — Multi-Pass DLSS 5, JonXL H3 Upscale in Open-H3, Batch Folders in Chains, Folder and File Picker Buttons on Every Path",
        """
**Run DLSS 5 two or three times for a much stronger photoreal look, upscale blurry footage with JonXL's MiniMax H3 upscale workflow, push a whole folder of videos and images through a chain, and pick every folder and file with a 📂 / 📄 button.**

**JonXL H3 Upscale: a second MiniMax H3 workflow in the Open-H3 tab**
- New **MiniMax H3 Workflow** switch next to "Auto Transfer Output to Input": Open-H3 Regenerate (the default, unchanged) or **JonXL H3 Upscale**, JonXL's MiniMax H3 image upscaler LoRA and workflow, built into the app's own Open-H3 engine.
- Every picture (an image, or every frame of a video) becomes the first frame of a 24-frame H3 clip that cross-fades from the blurry input into a sharp picture, and the last frame is the result: real detail pulled out of blurry, pixelated and compressed footage, up to 1920 px on the longest side (JonXL Max Size 1920 / 1536 / 1280). The picture keeps its exact aspect ratio (nothing stretched or cut off).
- Same recipe as JonXL's workflow: FL2VA + the `h3_upscale_1920` LoRA at strength 1, the fixed prompt "A static still blurry pixelated picture crossfades into a clear sharp detailed picture, 4psc4l3,", 20 steps er_sde on the beta57 schedule, CFG 1, Comfy Kitchen INT8 attention. Checked against JonXL's workflow run in ComfyUI: the same picture (45.7 dB).
- The LoRA (0.6 GB) downloads from our Hugging Face mirror only the first time you use the workflow; it needs only the selected DiT, text encoder and VAE besides. Switching to JonXL picks FL2VA INT8 ConvRot, the model of JonXL's workflow (W4A8 renders slightly darker with this LoRA).
- Works for images, videos, batch folders, Preview First Frame and Chains (the step card shows "JonXL H3 Upscale"). Every frame is a full 20-step H3 run, so try a short frame range of a video first.

**Multi-pass DLSS 5**
- New **Passes** setting (1-4) on the DLSS 5 tab: pass 2 renders DLSS 5 again on the result of pass 1, and so on. Every pass adds more real skin, hair, light and material detail, so 2-3 passes give the strongest look on game graphics, CGI and plastic-looking AI footage. Footage that already looks real can look overdone with several passes; start with 1-2 there.
- All passes run in one go: no re-encode between passes, and each pass keeps its own temporal history, so videos stay stable.
- In Chains you can also add DLSS 5 two or three times; the pipeline labels the steps pass 1/3, 2/3 and 3/3. Two new example chains: "⭐ DLSS 5 ×3 (multi-pass, strongest look)" and "⭐ SeedVR2 2x → DLSS 5 ×3".

**Chains: batch folders**
- New **📁 Batch processing** block right below Run Chain: enter a batch input folder and Run Chain processes every video and image in it, one after another. Results are saved in the batch output folder (empty = `<outputs>/<folder>_chain`) as `<name>_chain`, keeping the subfolders when "Include subfolders" is on.
- "Skip files that already have a result" (on by default) lets you press Run Chain again to continue an interrupted batch. A failing file never stops the batch, Cancel stops the whole batch, and Resume continues the files that were not finished. `chain_batch_report.json` in the output folder lists every file with its result.
- The block shows how many videos and images it found and how many are already done, and the input preview shows the batch files.

**📂 / 📄 picker buttons on every path field**
- Every folder and file path in the app (inputs, batch folders, output overrides, resume folders, model folders and files, the Global Settings folders) now has a picker button: 📂 opens a folder dialog, 📄 a file dialog filtered to videos, images or model files. It opens at the current path (or the last folder you picked), and a picked file refreshes the tab's preview exactly like a typed path.
- The dialog runs in its own small process, so it never freezes or crashes the app. On a server without a screen (RunPod) the buttons are hidden and typing paths works as before. The Windows installer now checks that Python's tkinter is present (part of the standard Python installer); the Massed Compute installer adds python3.12-tk.

**Chains fixes**
- The size shown after every step now follows each tab's Max Resolution and Open-H3's Target Resolution.
- An Open-H3 step with its own Scale now uses that scale (it switches Open-H3 to "Custom upscale factor"); before, the step's scale was ignored.
- The queue and the run's `chain.json` name an edited chain by its steps (for example "Custom chain: SeedVR2 → DLSS 5 ×3") instead of the preset it was loaded from.
- Step cards show the SparkVSR, LTX 2.5 and Open-H3 model instead of "tab defaults", and a DLSS 5 step shows its passes.
- Typing a batch folder and clicking away at once never leaves a stale file count behind.
""",
    ),
    (
        "V9.8 — 5 November 2026 — DLSS 5 Needs Less Than Half the VRAM and Runs Faster, Chains Get Input Preview, Quick Test, Resume and a Step Viewer",
        """
**DLSS 5 now renders 4K in about 3 GB of VRAM instead of 5-6 GB and is up to 40 % faster, with exactly the same picture. The Chains tab shows what you upload, tries a chain on a few seconds first, resumes a stopped run and lets you look at every step's result.**

**DLSS 5: less VRAM, faster, same result**
- The network's whole-frame attention stage built a table that grows with the square of the frame area, which set the memory floor at 4K. It is now computed a few heads at a time. The picture is bit-identical (checked on 1080p and 4K frames).
- DLSS 5 now picks the fastest work size for its execution mode instead of the largest one that fits in VRAM, and only goes smaller when VRAM is short.
- Measured on an RTX 5090 (network only): 1080p went from 2.7 GB to 0.8 GB and from 266 to 154 ms per frame; 4K from 5.0 GB to 2.6 GB and from about 1.1 to 0.7 s per frame.
- If a frame runs out of VRAM because another program took memory, DLSS 5 renders the same frame again with a smaller work size and then without CUDA Graphs, instead of stopping the run.

**Chains**
- **Preview:** the uploaded video or image appears next to the upload box with its size, frame rate, frame count and length. With several inputs, choose which one to preview. The pipeline shows the approximate size after every step for that input, and warns when the result would be larger than 8K.
- **Quick test:** run the chain on only the first 1-30 seconds of each video to try the settings before the full run.
- **Resume:** after a failed or cancelled run, "Resume unfinished run" continues from the step that stopped and reuses the finished steps. Each run folder's `chain.json` now records every step's result.
- **Step viewer:** after a run, view any step's result next to the step before it, not only the final video. "Open run folder" opens the run's folder.
- **Cancel** now also stops the step that is running, with that tab's own cancel, instead of waiting for it to finish.
- **Duplicate** a step with ⧉. On image inputs, video-only steps (LTX 2.5, RIFE) are skipped and the chain goes on.
- **Chain presets** store only the chain (its steps, order, On / Off and scale) in `presets/chains` and never change a tab's settings. App presets (Global Settings) now remember which chain preset you use and load it with the app preset; app presets saved before V9.8 load your newest saved chain.

**Fixed**
- SparkVSR: long 4K upscales no longer fail with "not enough memory" after every tile is done. Saving the video made a full-size copy of the whole clip in system RAM (30 GB for a 10-second 4K clip, next to two other full-size copies) and a 10-second 4x upscale to 4K ran out of memory on a 96 GB PC. The video is now converted a few frames at a time and the upscaled input is released before saving; the output is bit-identical.
""",
    ),
    (
        "V9.7 — 5 November 2026 — DLSS 5 Neural Rendering for Your Videos and Images, New Chains Tab",
        """
**NVIDIA's DLSS 5 neural renderer now runs on your own videos and images, on every RTX 30 / 40 / 50 GPU, and a new Chains tab runs several upscalers and effects one after another, for example SeedVR2 2x, then FlashVSR+ 2x, then DLSS 5.**

**New 💎 DLSS 5 tab**
- DLSS 5 re-renders each frame with photoreal lighting, skin, hair and material detail. It keeps the frame size; pair it with any upscaler (or a Chain) to also upscale.
- Videos keep their timing, frame count and audio, and stay stable over time: each frame reuses the previous rendered frame, moved along with the picture by optical flow, and the history resets by itself at scene cuts. On a test clip this cut the extra flicker DLSS 5 adds from 3.7 % to 0.7 %.
- Style profiles (Standard, Natural, Cinematic, Neutral), Intensity, Detail and Colour & lighting strength, Local tone / structure, Processing scale (render at up to 2x internally for finer detail), motion quality, batch folders and a Preview button that renders the first 48 frames.
- The model (292 MB) downloads automatically on the first run, also from `Models_Downloader.py` ("DLSS 5 Neural Rendering" or `--ensure-dlss5`).
- Runs as a PyTorch network on the GPU, so it needs no special driver, no extra program and no particular RTX generation. Checked on an RTX 3090 and an RTX 5090.
- Speed on an RTX 5090 (network only): about 20 frames/s at 960x540, 5.3 frames/s at 1080p and 1.1 frames/s at 4K, using up to 5.4 GB of VRAM at 4K. An RTX 3090 renders 1080p at about 1.6 frames/s.

**DLSS 5 after every upscale**
- Tick "Apply DLSS 5 automatically after every upscale" on the DLSS 5 tab and every finished video from SeedVR2, FlashVSR, SparkVSR, GAN, RTX Super Resolution, LTX 2.5 and Open-H3 also gets a `*_DLSS5.mp4`, rendered with the DLSS 5 tab settings. The upscaled file is kept. Global RIFE then interpolates the DLSS 5 frames.

**New 🔗 Chains tab**
- Click engines to build a pipeline: SeedVR2, FlashVSR+, SparkVSR, Image GAN, RTX Super Resolution, LTX 2.5, Open-H3, DLSS 5, Face Restoration and RIFE. Each step feeds its output to the next and uses the settings you set on its own tab (model, precision, VRAM Auto Tune, output codec); a step can set only its scale (or the RIFE frame-rate multiplier). Reorder, switch off or remove steps.
- Save chains as presets; four example chains are included. Run a chain on one video, several uploads or a whole folder. Every run gets its own folder under `outputs/chains` with each step's result, the final video and an Original vs Chain comparison video.
- Global post-processes (DLSS 5 after every upscale, Global RIFE, comparison videos) are switched off inside the steps, so nothing runs twice; add them as steps instead.

**Fixed**
- SeedVR2 no longer shows an "Out of VRAM" warning after an upscale that recovered by itself (its memory manager or the Auto Tune retry with the whole Save VRAM budget); the warning now appears only when the upscale really fails.
""",
    ),
    (
        "V9.6 — 2 November 2026 — Queued Jobs Keep the Settings of Their Click, Video Previews Keep Loading in Long Sessions",
        """
**Two fixes found by a full test of a fresh installation, which also confirmed that it installs every library the app needs.**

**Queued jobs use the settings you clicked with**
- When you clicked Upscale while other jobs were running, the job read its settings only after two small preparation requests (the Auto Tune preview and the "Auto Transfer Output to Input" check) came back from the busy app. Anything you changed in those seconds, such as the model or precision for your next job, could end up in the job you had just queued.
- The job now reads its settings at the moment you click, on every tab (SeedVR2, FlashVSR, Open-H3, LTX 2.5, SparkVSR, RTX Super Resolution and GAN). The Auto Tune preview still appears right after the click, and each job still re-tunes itself when it actually starts.

**Video previews no longer stop loading**
- After many runs in one browser tab, Chrome refused to open more videos, so previews, outputs and the comparison slider stayed on "Loading videos..." until the page was refreshed. Videos that leave the page now release their player at once, so long sessions keep working.
""",
    ),
    (
        "V9.5 — 2 November 2026 — W4A8 Default for LTX 2.5 and MiniMax Open-H3, FlashVSR INT8 Everywhere and Faster FlashVSR+, LTX Frame-Rate Fix",
        """
**The smallest models are now the defaults where they lose nothing measurable: W4A8 for SeedVR2, LTX 2.5 and MiniMax Open-H3, and INT8 ConvRot for FlashVSR on every platform. W4A8 keeps the quality of INT8 (within about 0.1 dB), downloads 30-40 % less and is 20-40 % faster whenever INT8 does not fit in your GPU's memory. FlashVSR+ is 12-15 % faster.**

How it was measured: test videos (a detailed 1080p robot clip, fast-motion ski footage and a face clip) were downscaled, upscaled back and compared with the originals (PSNR in dB, higher = closer to the original) on an RTX 5090; 24 GB and 16 GB GPUs were simulated by limiting the usable memory.

**New defaults**
- **LTX 2.5: Distilled INT4 W4A8 ConvRot** (15.4 GB instead of 21.9 GB). Within 0.04-0.11 dB of INT8 on every test clip, with the same flicker. At 1080p even a 32 GB GPU could not keep the whole INT8 model in memory, so W4A8 is faster there too: the sampling step took **40.2 s instead of 50.6 s** on an RTX 5090 and **46.0 s instead of 58.7 s** on a 24 GB GPU.
- **MiniMax Open-H3: the new FL2VA Pruned W4A8** (12.5 GB instead of 21.0 GB, about 7.5 GB less VRAM). Within 0.16 dB of INT8 either way, equally sharp in the zoomed crops. On 24 GB and 16 GB GPUs it keeps more (or all) of the model in memory: **14.5 s instead of 23.0 s** (24 GB) and **18-21 s instead of 29-34 s** (16 GB) for the same refine. The W6A8 option stays available (same quality as INT8, 4 GB less VRAM) but is not faster.
- **SeedVR2** keeps the 7B Sharp W4A8 default.
- **FlashVSR: INT8 ConvRot on every platform**, including FlashVSR+ Turbo and Base, which used to be forced to BF16. Same quality as BF16 (Turbo 37.248 vs 37.242 dB, Base 32.378 vs 32.382 dB), 0.5-1.1 GB less VRAM and slightly faster. Their INT8 files now download ready-made (1.7 GB) instead of being built on the first run.
- **GPUs older than RTX 30** (and CPU mode) switch LTX 2.5 and MiniMax from W4A8 / W6A8 to INT8 ConvRot automatically, with a note in the log, the same way SeedVR2 already did. Your saved presets keep the model they name.
- Auto Tune plans with the model you select: SeedVR2 and FlashVSR from measured sizes per precision, LTX 2.5 and MiniMax from the real size of the loaded model.

**FlashVSR+**
- **12-15 % faster:** its position encoding ran in slow double precision on every attention layer; it now uses a fused GPU kernel (Turbo 1080p: 13.1-14.1 s to 11.3-12.3 s) with the same picture.
- INT8 ConvRot no longer carries an error-correction branch that cost speed and added no measurable quality, so INT8 is now about 4 % faster than BF16 (it used to be no faster).
- Auto Tune no longer reserves 1.1 GB of extra VRAM for FlashVSR+ INT8 (it now uses less than BF16).

**LTX 2.5**
- **Fixed: finished upscales were rejected and deleted** when a video's nominal and average frame rates differed slightly (for example 29.87 vs 29.97 fps). Rendering and the final check now use the same source timing. If something else stops a result from being accepted, the rendered video is kept in the run's `recovery` folder instead of deleted, also during batch cleanup.

**Model downloads**
- Every precision the app offers downloads automatically on first use from public sources that need no login. MiniMax files are taken from the SECourses mirror first; the W4A8 file is checked against its exact size and checksum.
- Fixed: SeedVR2 FP8 models with INT8 ConvRot failed on a fresh install; the source model now downloads and the INT8 file is built on first use.
- `Models_Downloader.py`: the LTX 2.5 and Open-H3 default sets now download the W4A8 models (about 33 GB and 47 GB), the FlashVSR+ entries the INT8 files, and "ALL LTX 2.5 files" no longer fails on three files the app does not use.

**INT8 ConvRot**
- Slightly faster on small layers (FlashVSR, SparkVSR, SeedVR2 INT8): a per-layer correction matrix is now prepared once instead of on every step, with identical output.
""",
    ),
    (
        "V9.4 — 1 November 2026 — SeedVR2 W4A8 Default and 2x Faster INT8, New FlashVSR+ Turbo, Faster RIFE Without Ghost Frames, Faster GAN Video, LTX 2.5 Sparse Attention",
        """
**Every upscaler was checked against its upstream project and its forks, and everything that measured better on real test videos is in. SeedVR2's main model step now runs about 2x faster than BF16 with both the new default W4A8 model and the fixed INT8 ConvRot path, at the same quality, the new FlashVSR+ Turbo beats FlashVSR v1.1 on every test clip, RIFE is up to 2.4x faster with no ghost frames at scene cuts, and GAN video upscaling is 2.5-3.6x faster.**

How it was measured: test videos (a detailed 1080p robot clip, fast-motion ski footage and a face clip) were downscaled, upscaled back and compared with the originals (PSNR in dB, higher = closer to the original), on an RTX 5090 and an RTX 3090.

**Run `Windows_Install_and_Update.bat` once after updating:** it installs the new GAN architecture library and NVIDIA's video effects package 0.2.0.0 (needed for Frame Generation, 10-bit and HDR10 in the RTX tab).

**SeedVR2**
- **New default model: 7B Sharp W4A8.** The same quality as 7B Sharp INT8 ConvRot (within 0.1 dB on every test, and the zoomed crops look the same), the smallest download (4.8 GB) and the least VRAM (4.5 GB of model weights instead of 8.1 GB). Main model step:
  - RTX 5090, 720p ski clip: **4.5 s** (plain BF16: 9.0 s).
  - RTX 3090, same settings for every model: 720p ski **11.5 s**, 1080p robot **20.5 s** (BF16: 28.4 s and 50.2 s).
  - New preset in the model downloader; the app also fetches it when first needed. Saved presets keep the model they name.
- **RTX 20-series, GTX 16 and older cards** run 7B Sharp INT8 ConvRot automatically (W4A8 needs an RTX 30-series card or newer), with a note in the log, and Auto Tune plans for that model.
- **INT8 ConvRot is about 2x faster than BF16 again.** It had been running its layers in FP32, which made it slower than BF16 and used about 35% more memory; it now uses the fast INT8 kernels. RTX 5090 main model step: 13.1 → **4.7 s** on the 720p ski clip and 28.6 → **10.0 s** on the 1080p robot clip (BF16: 9.0 s and 28.1 s), with the same quality.
- **Auto Tune is accurate for every model.** Re-measured on an RTX 5090 and an RTX 3090, every phase estimate is now within -2% to +6% of the measured peak. It also counts the colour correction after decoding, frees cached memory before it and holds the run to its estimate, so the GPU memory a run uses matches the number it shows (test in the app: 17.6 GB used for a 19.7 GB estimate; before, 22.1 GB for 19.5 GB). Before, it also overestimated BF16 and W4A8 by 17-25% (so 24 GB cards got BlockSwap they did not need) and underestimated 3B INT8 on 24 GB cards by 14% (that run ran out of VRAM).
- **Faster and lighter with every model:** the main model needs 0.7-3.6 GB less VRAM and its step is 22-29% faster, video VAE encode and decode are 43-47% faster (about a quarter off a whole 1080p run), and single images need 31-49% less VAE memory.
- **GPUs without BF16 support (RTX 20-series, GTX 16):** the 7B model came out visibly distorted (25.7 dB); it now matches the BF16 result (34.1 dB vs 34.4 dB).
- **Fixed: Prepend Frames left the extra frames in the output.** A 61-frame video came out 65 frames long with the picture 4 frames late (133 ms out of sync with the sound); it is now exact, and a 900-frame video in 6 chunks keeps 900 frames and 30.000 s.
- **Fixed:** with torch.compile and model caching on, the second file of a batch crashed; seeds above 4,294,967,295 and negative seeds crashed (the same seed still gives the same picture).
- In-process mode finishes the 1080p test in 67 s instead of 115 s, and with BlockSwap it now frees its RAM after decoding (11.2 → 2.8 GB).

**FlashVSR (the tab was "FlashVSR+")**
- **New models: FlashVSR+ Turbo and FlashVSR+ Base** (the official FlashVSR+ release), next to FlashVSR v1.1. **Turbo beat v1.1 on all three test clips:** robot 31.6 vs 30.2 dB, ski 27.1 vs 25.7 dB, faces 26.0 vs 23.5 dB, with less flicker. Its colours stay true to the source (v1.1 shifts them darker and more saturated) and faces are sharper. 4K output runs without tiles in 64 s instead of 113 s, and a 900-frame clip runs in one pass in 98 s instead of 118 s. Base scores below Turbo and needs more VRAM.
- Each FlashVSR+ model downloads once (about 3.4 GB, BF16) and Auto Tune plans for it from measured VRAM. FlashVSR+ always runs in BF16: INT8 ConvRot was about 40% slower with no VRAM saving.
- **Fixed: tiled runs had a black 1-pixel border** on every frame.
- **Fixed: videos were saved about half a brightness level too dark** (pixel values were cut off instead of rounded); rounding adds +0.3 to +0.4 dB.
- FlashVSR v1.1 needs about 1 GB less VRAM at 1080p, with the same output.

**RIFE frame interpolation and RTX**
- **RIFE is up to 2.4x faster:** 81 → 192 fps at 320x180 (2x), +18% at 1080p (2x), +26% at 4K (2x), +34% at 1080p (4x).
- **No more ghost frames at scene cuts,** also when the RIFE tab splits a long video into chunks. The new cut detection found all 22 cuts in the test clips (the old one found 1); on a long clip with 12 cuts, blended ghost frames went from 11 to 0, and chunked runs now give exactly the right number of frames.
- **Fixed: the frames RIFE generates were slightly darker,** so 2x videos pulsed in brightness every other frame.
- **Exact frame rates:** 24, 23.976 or 25 fps → 60 fps now gives exactly the right number of frames with 65-86% less judder; the sound stays in sync and the video is encoded once instead of twice.
- RIFE no longer replaces real frames with blends in slow shots (+0.9 to +1.1 dB on those frames).
- **New: NVIDIA Frame Generation** as an interpolation choice in the RIFE tab and Global RIFE (RTX 40 and 50 series): 2-4x faster than RIFE up to 1080p at about the same quality.
- **RTX Super Resolution: fixed colours.** HD videos were decoded with the wrong colour standard (robot test 38.1 → 44.0 dB), and outputs now keep the source's colour information. 5.6% faster at 1080p → 4K.
- **New RTX options:** 10-bit output (8x less banding in gradients, 30-50% slower), HDR10 output with TrueHDR and VSR strength. The STREAMING levels now say in the UI that they sharpen strongly (they score far below High Bitrate on clean video). RIFE also gets duplicate-frame removal for animation.

**LTX 2.5**
- **Sparse attention on by default:** sampling is 9-14% faster with the same picture.
- **Sharper last frames:** extra tail frames improve the end of each video (+0.4 dB robot, +1.7 dB faces); they are skipped when they would need a whole extra chunk.
- **Faster decoding:** Auto Tune now plans the tail frames too and picks larger VAE decode tiles when VRAM allows, and the last chunk frees the transformer before decoding, so the large tiles fit (241-frame test: 8.5 → 6.4 s). If VRAM still runs short, it falls back to 512 px tiles.
- **Fixed: the NVFP4 model download was an older checkpoint.** It now matches the official release, and the app replaces an old copy automatically. Every LTX file is now checked by size and checksum, so outdated copies (including an outdated Gemma text encoder) are replaced.
- **New optional Refine: Restore** for old, faded or sepia archive footage: it restores detail and colour. It changes the colours of normal colour video, so leave it off there.

**GAN upscalers and face restoration**
- **Video upscaling is 2.5-3.6x faster:** models that support it now run in FP16 automatically.
- **Tiled upscaling fixed:** 1080p → 4K took 220 s and now takes 17 s, without seams.
- **More architectures load and run:** SMoSR, SST, DIS, FIGSR, GaterV3, GateRv2 and more, and UltraSharpV2 runs 2.8x faster in FP32. Output pixel values are rounded instead of cut off.
- **Face restoration:** "auto" now uses GFPGAN v1.4 (+0.9 to +1.7 dB on faces over v1.3), **CodeFormer really runs now**, and the strength slider really blends. Videos were restored twice, which made hair edges glow; they are now restored once and look natural (face test: 79.7 → 12.9 s).
""",
    ),
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
