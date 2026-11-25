"""Read EXR into the LTX ACEScct working space and write EXR plus HLG masters."""

import re
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .video_io import find_ffmpeg, _format_fps_rate


def read_exr_sequence(path: str, color_space: str) -> torch.Tensor:
    import OpenImageIO as oiio
    from ltx_core.color.primaries import Primaries
    from ltx_core.hdr import to_acescct_working_space

    source = Path(path)
    files = list(source.glob("*.exr")) if source.is_dir() else [source]
    files.sort(key=lambda file: [int(s) if s.isdecimal() else s.casefold() for s in re.split(r"(\d+)", file.name)])
    if not files or any(not file.is_file() or file.suffix.lower() != ".exr" for file in files):
        raise ValueError("Choose an EXR file or a directory of EXR frames.")
    primaries = {"ACEScg": Primaries.AP1, "Linear Rec.709": Primaries.REC709, "ACES2065-1": Primaries.AP0}
    frames = []
    for frame in files:
        buf = oiio.ImageBuf(str(frame))
        pixels = buf.get_pixels(oiio.FLOAT)
        if pixels is None or pixels.ndim != 3 or pixels.shape[-1] < 3:
            raise RuntimeError(f"Could not load an RGB EXR frame from {frame}")
        rgb = torch.from_numpy(np.ascontiguousarray(pixels[..., :3])).float()
        if not torch.isfinite(rgb).all():
            raise ValueError(f"Non-finite HDR values in {frame}")
        if color_space == "ACEScct":
            frames.append(rgb.clamp(0, 1))
        else:
            chw = rgb.permute(2, 0, 1).unsqueeze(0)
            frames.append(to_acescct_working_space(chw, primaries[color_space])[0].permute(1, 2, 0))
    return torch.stack(frames)


def anchor_detail(restored: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Keep source color and broad tone in ACEScct and transfer bounded luminance detail."""
    target = tuple(restored.shape[1:3])
    reference = F.interpolate(source.movedim(-1, 1).float(), size=target, mode="bilinear", align_corners=False)
    difference = restored.movedim(-1, 1).float().mean(dim=1, keepdim=True) - reference.mean(dim=1, keepdim=True)
    smooth = F.avg_pool2d(F.pad(difference, (4, 4, 4, 4), mode="replicate"), 9, stride=1)
    detail = (difference - smooth).clamp_(-0.0075, 0.0075)
    return (reference + detail).clamp_(0, 1).movedim(1, -1)


class HdrWriter:
    def __init__(self, output, width, height, fps, crf, source=None):
        from ltx_core.color.hlg import HlgGpuConverter
        from ltx_core.color.primaries import Primaries

        self.width, self.height = width, height
        self.frames_written = 0
        self.output = Path(output)
        self.source = source
        self.exr = Path(output).with_name(Path(output).stem + "_ACEScg_EXR")
        self.exr.mkdir(parents=True, exist_ok=False)
        self.convert_hlg = HlgGpuConverter.create(Primaries.AP1, device=torch.device("cpu"))
        self.process = subprocess.Popen(
            [find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p10le",
             "-s", f"{width}x{height}", "-r", _format_fps_rate(fps), "-i", "pipe:0",
             "-c:v", "libx265", "-crf", str(int(crf)), "-preset", "fast", "-pix_fmt", "yuv420p10le",
             "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc",
             "-x265-params", "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited",
             "-color_range", "tv", "-tag:v", "hvc1", "-movflags", "+faststart+write_colr", str(output)],
            stdin=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def write(self, frames: torch.Tensor):
        import OpenImageIO as oiio
        from ltx_core.color.primaries import Primaries
        from ltx_core.hdr import to_hdr_linear

        if self.source is not None:
            end = self.frames_written + frames.shape[0]
            frames = anchor_detail(frames, self.source[self.frames_written:end])
        rgb = to_hdr_linear(frames.float().permute(3, 0, 1, 2).unsqueeze(0), out_primaries=Primaries.AP1)
        linear = rgb[0].permute(1, 2, 3, 0).contiguous().cpu()
        y, u, v = self.convert_hlg(linear)
        spec = oiio.ImageSpec(self.width, self.height, 3, oiio.HALF)
        spec.channelnames = ("R", "G", "B")
        spec.attribute("compression", "zip")
        spec.attribute("chromaticities", "float[8]", Primaries.AP1.exr_chromaticities)
        spec.attribute("colorSpace", "linear ACEScg")
        for index in range(linear.shape[0]):
            path = str(self.exr / f"frame_{self.frames_written:06d}.exr")
            out = oiio.ImageOutput.create(path)
            if not out or not out.open(path, spec):
                raise RuntimeError(f"Could not open EXR output {path}")
            try:
                if not out.write_image(linear[index].numpy()):
                    raise RuntimeError(out.geterror())
            finally:
                out.close()
            for plane in (y, u, v):
                self.process.stdin.write(plane[index].contiguous().numpy().astype("<u2", copy=False).tobytes())
            self.frames_written += 1

    def close(self):
        self.process.stdin.close()
        code = self.process.wait()
        if code:
            raise RuntimeError(f"HLG encoding exited with code {code}.")
        self._set_mp4_color()

    def _set_mp4_color(self):
        # Align the MP4 container's nclx tags with the HEVC stream's BT.2020 HLG VUI.
        with self.output.open("r+b") as handle:
            length = self.output.stat().st_size
            while handle.tell() < length:
                offset = handle.tell()
                head = handle.read(8)
                if len(head) != 8:
                    break
                size = int.from_bytes(head[:4], "big")
                header_size = 8
                if size == 1:
                    size = int.from_bytes(handle.read(8), "big")
                    header_size = 16
                if size < header_size or offset + size > length:
                    raise RuntimeError("Invalid MP4 box while applying the HLG container tags.")
                if head[4:] == b"moov":
                    data = handle.read(size - header_size)
                    match = data.find(b"\x00\x00\x00\x13colrnclx")
                    if match < 0 or data.find(b"\x00\x00\x00\x13colrnclx", match + 1) >= 0:
                        raise RuntimeError("Expected one nclx color box in the HLG master.")
                    handle.seek(offset + header_size + match + 12)
                    handle.write(bytes((0, 9, 0, 18, 0, 9, 0)))
                    return
                handle.seek(offset + size)
        raise RuntimeError("Missing MP4 movie box in the HLG master.")
