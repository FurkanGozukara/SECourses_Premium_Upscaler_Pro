"""ReCaVSR with INT8 weights, bounded frame IO and the app's encoder."""

from itertools import islice
from importlib.util import module_from_spec, spec_from_file_location
from types import SimpleNamespace
import sys

import numpy as np
import torch

from experimental_vsr_inference import APP, MODELS, prepare_assets
from shared.video_io import FfmpegWriter, iter_video_frames, probe_video


def assets_for(args):
    from huggingface_hub import hf_hub_download

    source, weights = prepare_assets(args.model, args.assets)
    _, _, repo, revision, folder, _ = MODELS["ReCaVSR"]
    stock = args.assets / folder
    for name in ("prompt.safetensors", "flashdecoder.safetensors"):
        if not (stock / name).is_file():
            hf_hub_download(repo, name, revision=revision, local_dir=stock)
    return source, (weights / "recavsr/transformer_int8_convrot.safetensors",
                    stock / "flashdecoder.safetensors", stock / "prompt.safetensors")


@torch.inference_mode()
def restore_stream(args):
    source, files = assets_for(args)
    sys.path[:0] = [str(source), str(APP / "OpenH3")]
    from comfy import ops, utils
    from recavsr.dit import CONFIG, ReCaVSRTransformer
    from recavsr.flashdecoder import DecoderState, FlashDecoder
    from recavsr.session import DecoderStream, VideoStream, block_schedule, padded_size, prepare_block, wavelet_color_fix

    torch.cuda.reset_peak_memory_stats()
    device, host, dtype = torch.device("cuda:0"), torch.device("cpu"), torch.bfloat16
    state = utils.load_torch_file(str(files[0]))
    quant = utils.detect_layer_quantization(state, "")
    if quant is None:
        raise RuntimeError("ReCaVSR INT8 checkpoint is missing its quantization configuration.")
    operations = ops.pick_operations(dtype, dtype, load_device=device, model_config=SimpleNamespace(quant_config=quant))
    model = ReCaVSRTransformer(**CONFIG, device=host, dtype=dtype, operations=operations)
    model.load_state_dict(state, strict=True)
    del state
    model = model.eval().to(device)
    decoder = FlashDecoder(device=host, dtype=torch.float32, operations=ops.manual_cast)
    decoder.load_state_dict(utils.load_torch_file(str(files[1])), strict=True)
    decoder = decoder.eval().to(device)
    decoder.new_state = lambda: DecoderState(decoder)
    prompt = utils.load_torch_file(str(files[2]))["positive"]

    original = probe_video(args.input, count=False)
    count = min(original.frames, args.frames) if args.frames > 0 else original.frames
    size = (original.height * args.scale, original.width * args.scale)
    height, width = padded_size(*size)
    tokens = height // 32 * (width // 32)
    free_bytes = torch.cuda.mem_get_info(device)[0]
    activation_estimate = tokens * int(2.6 * 1024**2)
    if activation_estimate > free_bytes * .85:
        model.lq_proj.chunk = 4
        activation_estimate -= tokens * int(.66 * 1024**2)
    offload = activation_estimate > free_bytes * .85
    if offload:
        activation_estimate -= tokens * (66 * 2 * 3072 * 2)
    if activation_estimate > free_bytes:
        raise RuntimeError(f"ReCaVSR INT8 needs about {activation_estimate / 1024**3:.1f} GiB for this geometry after loading weights.")
    stream = VideoStream(model, prompt, height, width, 42, device, "exact", host if offload else None)
    spec = spec_from_file_location("app_wan_latent", APP / "LTX25/comfy/latent_formats.py")
    latent_formats = module_from_spec(spec)
    spec.loader.exec_module(latent_formats)
    decode = DecoderStream(decoder, latent_formats.Wan22(), device, dtype, "exact")
    audio = str(args.input) if original.has_audio else None
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FfmpegWriter(str(output), size[1], size[0], original.fps, audio_source=audio,
                          codec=args.codec, crf=14, preset="p4" if args.codec == "h264_nvenc" else "ultrafast",
                          audio_duration_s=float(count / original.fps) if count < original.frames else None,
                          video_args="-c:v libx264 -crf 14 -preset ultrafast -threads 2 -pix_fmt yuv420p"
                          if args.codec == "libx264" else "")
    frames = iter_video_frames(args.input)
    try:
        for start, valid, block in block_schedule(count):
            inputs = np.stack(list(islice(frames, valid)))
            lq = prepare_block(inputs, 0, valid, block, size, device)
            latents = stream.step(lq)
            done = 0
            for result in decode.step(latents, lq):
                take = min(result.shape[2], valid - done)
                if take <= 0:
                    break
                rgb = result[0, :, :take, :size[0], :size[1]].transpose(0, 1)
                corrected = wavelet_color_fix(rgb, inputs[done:done + take])
                pixels = corrected.mul_(255).round_().to(torch.uint8).permute(0, 2, 3, 1)
                writer.write(pixels.contiguous().cpu())
                done += take
            if done != valid:
                raise RuntimeError("ReCaVSR decoder returned an incomplete frame block.")
            print(f"FRAME_PROGRESS {start + done}/{count} frames", flush=True)
        writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        frames.close()
    print(f"ReCaVSR INT8 streaming peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB", flush=True)
    print(f"OUTPUT_PATH: {output}", flush=True)
