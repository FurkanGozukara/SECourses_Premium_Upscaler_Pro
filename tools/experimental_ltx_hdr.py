"""Run the app's LTX 2.5 refiner with native HDR input and output."""

import sys

from experimental_vsr_inference import APP, ensure_dependencies, ensure_source


REVISION = "c3a7efb18256cba2d76cc32f5b47811d7c9ad642"


def restore_hdr(args):
    from shared.ltx25_constants import (
        LTX25_AUDIO_VAE_FILE, LTX25_DISTILLED_INT8, LTX25_MODELS_DIRNAME,
        LTX25_REFINE_PROMPTS, LTX25_RESTORE_LORA_FILE, LTX25_TE_FILES, LTX25_TE_INT8,
        LTX25_TRANSFORMER_FILES, LTX25_VAE_CONV, LTX25_VAE_FILES,
    )

    if args.scale != 2:
        raise ValueError("Choose 2× for LTX 2.5 HDR Restore.")
    ltx_source = args.assets / "source" / f"ltx-2-{REVISION}"
    ensure_source("Lightricks/LTX-2", REVISION, ltx_source, "packages/ltx-core/src/ltx_core/hdr.py")
    ensure_dependencies(args.assets, {"OpenImageIO": "openimageio==3.2.2.0", "colour": "colour-science==0.4.7"})
    sys.path.insert(0, str(ltx_source / "packages/ltx-core/src"))
    sys.path.insert(0, str(APP / "LTX25"))
    from ltx25_engine.pipeline import Ltx25Pipeline, Ltx25Progress, Ltx25Settings

    models = APP / LTX25_MODELS_DIRNAME
    files = {
        "transformer_path": models / LTX25_TRANSFORMER_FILES[LTX25_DISTILLED_INT8],
        "text_encoder_path": models / LTX25_TE_FILES[LTX25_TE_INT8],
        "video_vae_path": models / LTX25_VAE_FILES[LTX25_VAE_CONV],
        "audio_vae_path": models / LTX25_AUDIO_VAE_FILE,
        "ic_lora_path": models / LTX25_RESTORE_LORA_FILE,
    }
    if any(not file.is_file() for file in files.values()):
        from shared.model_downloads import ensure_ltx25_model

        ok, reason = ensure_ltx25_model(APP, LTX25_DISTILLED_INT8, LTX25_TE_INT8, LTX25_VAE_CONV,
                                        extra_files=(LTX25_RESTORE_LORA_FILE,))
        if not ok:
            raise RuntimeError(reason)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = Ltx25Settings(**{key: str(path) for key, path in files.items()},
                             input_path=str(args.input), output_path=str(output), fps_override=args.hdr_fps,
                             end_frame=args.frames - 1, hdr_input_space=args.hdr_space, mode="refine",
                             spatial_tiling="auto", spatial_tile_width=960, spatial_tile_height=544,
                             spatial_tile_blend="gaussian", steps=8, preclean_lora_path="",
                             positive_prompt=LTX25_REFINE_PROMPTS["Restore 1.0"])
    Ltx25Pipeline(settings, Ltx25Progress(on_line=lambda line: print(line, flush=True))).run()
    print(f"ACEScg EXR directory: {output.with_name(output.stem + '_ACEScg_EXR')}", flush=True)
    print(f"OUTPUT_PATH: {output}", flush=True)
