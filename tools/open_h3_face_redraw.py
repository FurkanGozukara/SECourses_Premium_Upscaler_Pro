"""Re-draw tracked small faces with the selected Open-H3 model and blend them into the export."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from itertools import islice
from pathlib import Path
import tempfile

import numpy as np
import torch

from experimental_vsr_inference import ensure_source
from open_h3_engine.pipeline import OpenH3Pipeline, run_many
from shared.open_h3_constants import OPENH3_UPSAMPLER_REVISION
from shared.video_io import FfmpegWriter, iter_video_frames, probe_video


def _batches(frames, batch_size=8):
    frames = iter(frames)
    while group := list(islice(frames, batch_size)):
        yield torch.from_numpy(np.stack(group))


def apply_small_face_redraw(pipeline, output: str, assets: Path) -> str:
    revision = OPENH3_UPSAMPLER_REVISION
    source = assets / f"h3-upsampler-{revision}"
    ensure_source("sandpies/ComfyUI-H3-Video-Upsampler", revision, source, "face_pass.py")
    spec = spec_from_file_location("h3_small_face_pass", source / "face_pass.py")
    face = module_from_spec(spec)
    spec.loader.exec_module(face)

    cfg = pipeline.s
    indices, fps, width, height, has_audio, is_image = pipeline._timeline()
    if is_image:
        return output
    info = probe_video(output)
    if info.frames != len(indices):
        raise RuntimeError("Open-H3 face pass requires a source-aligned frame count.")
    sx, sy = info.width / width, info.height / height

    def source_batches():
        for batch in _batches(pipeline._iter_source(indices)):
            yield batch.float().div_(255)

    detections = face.detect_batches(source_batches())
    if detections is None:
        raise RuntimeError("Open-H3 small-face detection could not initialize.")
    jobs = face.select(detections, len(indices), (sx + sy) / 2)
    if not jobs:
        pipeline.line("Small-face pass found no eligible tracks.")
        return output
    crops = face.crops_from_batches(jobs, source_batches())
    source_rate = probe_video(cfg.input_path, count=False).fps
    use_guide_audio = cfg.audio_mode == "source" and has_audio

    path = Path(output)
    with tempfile.TemporaryDirectory(prefix="openh3-faces-", dir=path.parent) as work:
        folder = Path(work)
        renders = []
        for number, (job, crop) in enumerate(zip(jobs, crops), 1):
            crop_path = str(folder / f"input-{number}.mp4")
            guide_audio = cfg.input_path if use_guide_audio else None
            with_writer = FfmpegWriter(crop_path, face.CROP_IN, face.CROP_IN, fps,
                                      audio_source=guide_audio, crf=0, preset="ultrafast",
                                      audio_offset_s=float(indices[job["f0"]] / source_rate),
                                      audio_duration_s=float(job["m"] / fps))
            try:
                for part in crop.split(8):
                    with_writer.write(part)
                with_writer.close()
            except BaseException:
                with_writer.abort()
                raise
            settings = replace(cfg, input_path=crop_path, output_path=str(folder / f"face-{number}.mp4"),
                               target_width=face.CROP_OUT, target_height=face.CROP_OUT,
                               draft_size="native", start_frame=0, end_frame=-1, fps_mode="source",
                               refine_tile_px=0, codec="libx264", preset="ultrafast", crf=14,
                               video_args="", keep_audio=False, keep_latents=False)
            renders.append(OpenH3Pipeline(settings, pipeline.progress))
        del crops
        outcomes = run_many(renders, pipeline.progress)
        for number, (job, (file, error)) in enumerate(zip(jobs, outcomes), 1):
            if error or not file:
                raise RuntimeError(f"Small-face track {number} did not render: {error}")
            pixels = np.stack(list(iter_video_frames(file)))
            if len(pixels) != job["m"]:
                raise RuntimeError(f"Small-face track {number} returned a different frame count.")
            job["out"] = torch.from_numpy(pixels)
        blended = str(folder / path.name)
        writer = FfmpegWriter(blended, info.width, info.height, info.fps,
                              audio_source=output if info.has_audio else None,
                              codec=cfg.codec, crf=cfg.crf, preset=cfg.preset, pixel_format=cfg.pixel_format,
                              video_args=cfg.video_args, audio_duration_s=float(info.frames / info.fps))
        offset = 0
        try:
            for frames in _batches(iter_video_frames(output)):
                writer.write(face.paste_jobs(frames, offset, jobs, sx, sy))
                offset += frames.shape[0]
            writer.close()
        except BaseException:
            writer.abort()
            raise
        Path(blended).replace(path)
    pipeline.line(f"Small-face pass re-drew {len(jobs)} track(s).")
    return output
