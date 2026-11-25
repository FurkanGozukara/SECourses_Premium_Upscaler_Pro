"""Preview opt-in SwiftVR and ReCaVSR video restorations."""

from collections import deque
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import gradio as gr

from shared.gpu_utils import get_global_gpu_override
from shared.ltx25_runner import _build_env, _resolve_python_executable, _stream_process
from shared.open_h3_runner import _popen, _resolve_device
from shared.output_run_manager import allocate_sequential_run_dir
from shared.path_utils import normalize_path
from shared.processing_queue import get_processing_queue_manager, resolve_queue_gpu_resources
from shared.queue_state import snapshot_global_settings, snapshot_queue_state

IMAGE_MODELS = ("VOSR 2.0", "HEART", "Row-banded ESRGAN")
HDR_MODEL = "LTX 2.5 HDR Restore"


def experimental_vsr_tab(base_dir: Path, output_dir: Path, shared_state, global_settings):
    with gr.Row():
        model = gr.Dropdown(
            label="Restoration Model",
            choices=["ReCaVSR", "ReCaVSR INT8", "SwiftVR", "SMoVSR", *IMAGE_MODELS, HDR_MODEL],
            value="ReCaVSR",
            info="ReCaVSR, SwiftVR and SMoVSR restore videos. VOSR 2.0, HEART and ESRGAN restore still images.",
        )
        scale = gr.Radio(label="Scale", choices=[2, 4], value=2)
    source = gr.Textbox(label="Source Media Path", placeholder="C:/path/to/video.mp4 or image.png")
    frame_limit = gr.Number(label="Preview Frames (0 = full clip)", precision=0, minimum=0, value=0)
    checkpoint = gr.Textbox(label="Trained SMoVSR Checkpoint Path", visible=False,
                            info="Select the safetensors file for your trained checkpoint.")
    recavsr_codec = gr.Radio(label="ReCaVSR INT8 Encoder",
                             choices=[("CPU H.264", "libx264"), ("NVIDIA H.264", "h264_nvenc")],
                             value="libx264", visible=False)
    with gr.Row(visible=False) as hdr_controls:
        hdr_space = gr.Dropdown(label="EXR Color Space", choices=["ACEScg", "ACEScct", "Linear Rec.709", "ACES2065-1"],
                                value="ACEScg")
        hdr_fps = gr.Number(label="EXR Sequence Frame Rate", value=24.0, minimum=1.0)
    with gr.Row():
        render = gr.Button("Restore", variant="primary")
        stop = gr.Button("Cancel")
    log = gr.Textbox(label="Run Log", lines=8, max_lines=16, interactive=False)
    preview = gr.Video(label="Restored Video", interactive=False)
    still = gr.Image(label="Restored Image", interactive=False, visible=False)
    master = gr.File(label="HLG Master", interactive=False, visible=False)
    cancellation = gr.State(None)

    def run(model_name, factor, source_path, frame_count, checkpoint_path, codec, color_space, frame_rate, cancel, state):
        normalized = normalize_path(source_path or "")
        is_hdr = model_name == HDR_MODEL
        path = Path(normalized) if normalized else None
        if path is None or not (path.is_file() or (is_hdr and path.is_dir())):
            yield "Enter a source file or, for HDR restore, an EXR sequence directory.", None, None, None
            return
        is_image = model_name in IMAGE_MODELS
        ext = ".png" if is_image else ".mp4"
        state = snapshot_queue_state(state)
        globals_snapshot = snapshot_global_settings(global_settings)
        manager = get_processing_queue_manager()
        keys, label = resolve_queue_gpu_resources(state, globals_snapshot)
        ticket = manager.submit("Experimental Upscalers", model_name, resource_keys=keys, resource_label=label)
        messages = Queue()
        cancel_event = ticket.cancel_event
        launch = None

        def cancelled():
            if cancel is not None and cancel.event.is_set():
                cancel_event.set()
            return cancel_event.is_set()

        def worker():
            try:
                selected = get_global_gpu_override(state.get("seed_controls"), globals_snapshot)
                gpu, note = _resolve_device(selected.split(",")[0])
                if note:
                    messages.put(note)
                env = _build_env(base_dir, gpu, messages.put, tag=model_name)
                folder = allocate_sequential_run_dir(output_dir / "experimental_upscalers")
                target = folder / f"{path.stem}-{model_name}-{int(factor)}x{ext}"
                command = [
                    _resolve_python_executable(base_dir), "-u", str(base_dir / "tools/experimental_vsr_inference.py"),
                    "--model", model_name, "--input", str(path), "--output", str(target),
                    "--scale", str(int(factor)), "--frames", str(int(frame_count or 0)),
                ]
                if model_name == "SMoVSR" and checkpoint_path:
                    command += ["--checkpoint", normalize_path(checkpoint_path)]
                if model_name == "ReCaVSR INT8":
                    command += ["--codec", codec]
                if is_hdr:
                    command += ["--hdr-space", str(color_space), "--hdr-fps", str(frame_rate)]
                process = _popen(command, base_dir, env)
                code, reported, cancelled = _stream_process(process, messages.put, cancel_event, None,
                                                            tag=model_name, label=model_name)
                if cancelled:
                    messages.put(("Cancelled", None))
                elif code == 0 and reported and target.is_file():
                    messages.put(("Finished", str(target)))
                else:
                    messages.put((f"{model_name} exited with code {code}.", None))
            except Exception as exc:
                messages.put((str(exc), None))

        lines = deque([f"Starting {model_name} {factor}×"], maxlen=80)
        try:
            if not globals_snapshot.get("queue_enabled", True) and not manager.is_active(ticket.job_id):
                yield "The selected GPU is busy.", None, None, None
                return
            while not ticket.start_event.wait(timeout=0.5):
                if cancelled():
                    yield "Cancelled", None, None, None
                    return
                yield "Waiting in the processing queue…", None, None, None
            if cancelled():
                yield "Cancelled", None, None, None
                return
            launch = Thread(target=worker, daemon=True)
            launch.start()
            yield lines[0], None, None, None
            while True:
                cancelled()
                try:
                    item = messages.get(timeout=0.5)
                except Empty:
                    continue
                if isinstance(item, tuple):
                    status, result = item
                    lines.append(status)
                    yield ("\n".join(lines), result if not (is_image or is_hdr) else None,
                           result if is_image else None, result if is_hdr else None)
                    return
                lines.append(item)
                yield "\n".join(lines), None, None, None
        finally:
            cancel_event.set()
            if launch is not None:
                launch.join()
            manager.cancel_waiting([ticket.job_id])
            manager.complete(ticket.job_id)

    class Cancellation:
        def __init__(self):
            self.event = Event()

        def __deepcopy__(self, memo):
            return self

    start_event = render.click(lambda: Cancellation(), outputs=[cancellation], queue=False, show_progress="hidden")
    start_event.then(
        run, [model, scale, source, frame_limit, checkpoint, recavsr_codec, hdr_space, hdr_fps, cancellation, shared_state],
        [log, preview, still, master],
        api_name="experimental_vsr_restore", concurrency_limit=32, concurrency_id="app_processing_queue",
    )
    stop.click(lambda token: token.event.set() if token is not None else None, cancellation, None, queue=False)
    def model_fields(name, factor):
        is_still, is_hdr = name in IMAGE_MODELS, name == HDR_MODEL
        return (gr.update(visible=not is_still),
                gr.update(visible=not (is_still or is_hdr), value=None),
                gr.update(visible=is_still, value=None), gr.update(visible=name == "SMoVSR"),
                gr.update(visible=name == "ReCaVSR INT8"),
                gr.update(visible=is_hdr), gr.update(visible=is_hdr, value=None),
                gr.update(value=2 if is_hdr else factor, interactive=not is_hdr))

    model.change(model_fields, [model, scale],
                 [frame_limit, preview, still, checkpoint, recavsr_codec, hdr_controls, master, scale],
                 queue=False)
