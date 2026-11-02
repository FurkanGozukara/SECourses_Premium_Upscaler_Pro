# LTX 2.5 output timing and recovery

Status: fixed and verified locally, 2026-11-02. Owner: Codex, Roland's V8.6 FPS report.
Base commit: `6e6a59e` (V9.4). Not committed, pushed, or added to a release zip.

## Coordination

The running INT8 audit is Claude session `cb206ddc-dfe1-4054-bff2-a73d7dc95fd6`
(`secourses-premium-upscaler-pro-v8-a2`). Its changes in
`shared/int8_convrot.py` and `docs/int8_convrot_v61.md` are outside this fix.
Other sessions own GPU 0; the current booking is `upscaler-pro-post-v94`.
This fix used CPU-only media tests and did not stop or restart their processes.

Reserved files: `shared/ltx25_runner.py`, `tools/test_ltx25_output_timing.py`, and
this note. The final cleanup audit also requires a small opt-in addition to
`shared/batch_output_cleanup.py` and its LTX call in
`shared/services/ltx25_service.py`: mixed batches must keep recovered files.
Direct cross-runtime messaging is not exposed in this Codex session; session
state, the booking ledger, and this shared note are the coordination channels.
The other app server on port 7896 (PID 85792) was left running. Only this task's
CPU-only launcher on port 7897 and named browser `ltx25-fps-fix` were stopped.

## Finding

The engine reads the source's average FPS through PyAV. The app normally uses
the nominal FPS when its difference from the average is small. The reported
rates reproduce the exact rejection in the current V9.4 validator. Its failure
handler then deletes the encoded video.

## Implementation and verification

- LTX selects the average rate even for small nominal/average differences. It
  passes that exact rate to the engine's existing writer FPS argument and checks
  the same rate afterward. No rounding and no increase in validation tolerance.
- A user FPS Override still converts the input to the requested CFR before
  inference; internally passing the chosen rate does not trigger preprocessing.
- Rejected output, nonzero engine exits, and unexpected validation exceptions
  retain encoded work under `<run folder>/recovery/`. Names avoid collisions,
  failed work is not returned as a successful output or resumable chunk, and the
  log gives the recovery path. If the move fails, the original file stays intact
  and the exact move error is logged.
- LTX batch cleanup opts into preserving recovery files, including a mixed batch
  with successful and failed items. Other cleanup callers retain their existing
  behavior. Chunk cleanup cannot erase recovery files at the run root.
- Frame count, real rate mismatch, duration, and zero-start checks remain active.

Before the fix, a real near-nominal VFR fixture was rejected with
`frame rate mismatch 29.870129870/29.970029970` and its encoded file was deleted.
The reported `30.084310352/29.970029970` case is also covered with probe metadata.

**40 tests passed** (22 new regressions and 18 existing checks), with
`CUDA_VISIBLE_DEVICES=-1`, using the project's venv:

```text
venv/Scripts/python.exe -m unittest tools.test_ltx25_output_timing tools.test_ltx25_plan tools.test_chunk_av_sync.IncompleteModelArtifactTests tools.test_chunk_av_sync.ChunkOutputTimingGuardTests tools.test_runtime_regressions.MediaFpsTests -v
```

Evidence: [test log](../temp/ltx25_fps_fix_tests.log),
[encoded fixtures and recovered files](../temp/ltx25_output_timing/021208e4e01f47c59f484f4af74f6a4c/),
[launcher log](../temp/ltx25_fps_ui_smoke.log), and
[LTX tab screenshot](../output/playwright/ltx25_fps_fix_ui.png).
The unchanged `Windows_Run_SECourses_Upscaler_Pro.bat` launched successfully;
the real LTX tab opened in an isolated browser with no console errors.
`git diff --check` passed for the fix.

The timing tests use real FFmpeg encoding, PyAV decoding, audio streams, final
validation, and filesystem recovery. They replace the GPU inference subprocess
with an already encoded result. No model weights or GPU kernels changed, and no
GPU synthesis was repeated or other session's jobs disturbed.

## Release note for the next update

LTX 2.5 no longer rejects a finished upscale when a video's nominal and average
frame rates differ slightly. Rendering and final validation now use the same
source timing. If another error prevents acceptance, the rendered video is kept
in the run's recovery folder instead of deleted, including during batch cleanup.
