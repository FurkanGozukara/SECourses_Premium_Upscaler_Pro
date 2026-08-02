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
        "V6.0 — 2 August 2026 — INT8 ConvRot Quantization, Modernized Interface, Gradio 6.22",
        """
**A massive quality-of-life and performance release.**

- **NEW: INT8 ConvRot quantization** — a third precision option next to BF16 and FP8 Scaled.
  - Better quality than FP8 Scaled (closer to the original BF16 model) while using about **half the VRAM of BF16**.
  - **Faster than BF16** on RTX 2000, 3000, 4000 and 5000 series GPUs (works on any NVIDIA GPU from GTX 1650 Super / RTX 2000 "Turing" and newer).
  - Available for **SparkVSR** (new "SparkVSR-int8-convrot" model choice) — the quantized model is generated automatically from the BF16 weights the first time you use it, exactly like the FP8 Scaled option.
  - Uses the same proven technology recently added to ComfyUI and our Musubi Trainer (group-wise Hadamard rotation + per-row MSE-optimized INT8).
- **Modernized interface on Gradio 6.22** — every button on every tab now has its own distinct color so you can find actions at a glance. Smoother tab switching, cleaner look in both themes.
- **Light theme fixed and polished** — the top navigation bar and health banner now look correct in light mode (dark mode unchanged and still the default).
- **Theme choice now sticks** — switching dark/light in Global Settings is remembered by your browser immediately, even before you save a preset and even after restarting the app.
- **NEW: this Version History tab** — the full release history of the app, right inside the interface.
- Internal health checks updated for the Gradio 6 interface engine.
- Library upgrade pass: everything re-tested against Torch 2.13 + CUDA 13 + Gradio 6.22 with fixes where needed.
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
