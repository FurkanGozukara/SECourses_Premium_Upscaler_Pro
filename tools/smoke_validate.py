from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.batch_processor import BatchJob, BatchProcessor
from shared.universal_preset import (
    TAB_CONFIGS,
    dict_to_values,
    get_all_defaults,
    update_shared_state_from_preset,
    values_to_dict,
)


def _check_universal_preset_caches() -> None:
    state = {"seed_controls": {}}
    preset = {
        "resolution": {"upscale_factor": 3.0, "max_target_resolution": 1024},
        "output": {"telemetry_enabled": False, "overwrite_existing_batch": True},
    }
    update_shared_state_from_preset(state, preset, preset_name="smoke")
    assert state["seed_controls"]["telemetry_enabled_val"] is False
    assert state["seed_controls"]["overwrite_existing_batch_val"] is True
    assert float(state["seed_controls"]["upscale_factor_val"]) == 3.0
    assert int(state["seed_controls"]["max_resolution_val"]) == 1024
    assert bool(state["seed_controls"].get("auto_detect_scenes", True)) is True
    assert bool(state["seed_controls"].get("frame_accurate_split", True)) is True

    # Auto-sync path (per-tab changes)
    from ui.universal_preset_section import sync_tab_to_shared_state
    from shared.services.output_service import OUTPUT_ORDER, output_defaults
    from shared.services.resolution_service import RESOLUTION_ORDER, resolution_defaults

    out_defaults = output_defaults(["default"])
    out_vals = [out_defaults[k] for k in OUTPUT_ORDER]
    out_vals[OUTPUT_ORDER.index("telemetry_enabled")] = False
    out_vals[OUTPUT_ORDER.index("overwrite_existing_batch")] = True
    state2 = {"seed_controls": {}}
    sync_tab_to_shared_state("output", out_vals, state2)
    assert state2["seed_controls"]["telemetry_enabled_val"] is False
    assert state2["seed_controls"]["overwrite_existing_batch_val"] is True
    assert state2["seed_controls"]["png_sequence_enabled_val"] is False

    res_defaults = resolution_defaults(["default"])
    assert bool(res_defaults.get("auto_chunk", True)) is True
    assert bool(res_defaults.get("auto_detect_scenes", True)) is True
    assert bool(res_defaults.get("frame_accurate_split", True)) is True
    assert float(res_defaults.get("chunk_overlap", 0.0) or 0.0) == 0.0
    assert float(res_defaults.get("min_scene_len", 0.0) or 0.0) == 1.0
    res_vals = [res_defaults[k] for k in RESOLUTION_ORDER]
    res_vals[RESOLUTION_ORDER.index("upscale_factor")] = 2.5
    res_vals[RESOLUTION_ORDER.index("max_target_resolution")] = 720
    sync_tab_to_shared_state("resolution", res_vals, state2)
    assert float(state2["seed_controls"]["upscale_factor_val"]) == 2.5
    assert int(state2["seed_controls"]["max_resolution_val"]) == 720
    assert bool(state2["seed_controls"].get("frame_accurate_split", True)) is True
    assert float(state2["seed_controls"].get("min_scene_len", 0.0) or 0.0) == 1.0

    res_vals_toggle = [res_defaults[k] for k in RESOLUTION_ORDER]
    res_vals_toggle[RESOLUTION_ORDER.index("frame_accurate_split")] = False
    sync_tab_to_shared_state("resolution", res_vals_toggle, state2)
    assert bool(state2["seed_controls"]["frame_accurate_split"]) is False

    res_vals_toggle2 = [res_defaults[k] for k in RESOLUTION_ORDER]
    res_vals_toggle2[RESOLUTION_ORDER.index("auto_detect_scenes")] = False
    sync_tab_to_shared_state("resolution", res_vals_toggle2, state2)
    assert bool(state2["seed_controls"]["auto_detect_scenes"]) is False

    # Auto-chunk enforces overlap=0.0 at the shared-state level.
    res_vals2 = [res_defaults[k] for k in RESOLUTION_ORDER]
    res_vals2[RESOLUTION_ORDER.index("auto_chunk")] = True
    res_vals2[RESOLUTION_ORDER.index("chunk_overlap")] = 3.0
    sync_tab_to_shared_state("resolution", res_vals2, state2)
    assert bool(state2["seed_controls"]["auto_chunk"]) is True
    assert float(state2["seed_controls"]["chunk_overlap_sec"]) == 0.0

    res_vals3 = [res_defaults[k] for k in RESOLUTION_ORDER]
    res_vals3[RESOLUTION_ORDER.index("auto_chunk")] = False
    res_vals3[RESOLUTION_ORDER.index("chunk_overlap")] = 1.5
    sync_tab_to_shared_state("resolution", res_vals3, state2)
    assert bool(state2["seed_controls"]["auto_chunk"]) is False
    assert float(state2["seed_controls"]["chunk_overlap_sec"]) == 1.5


def _check_universal_schema_roundtrip() -> None:
    defaults = get_all_defaults(ROOT)
    for tab_name, cfg in TAB_CONFIGS.items():
        order = cfg.get("order", [])
        tab_defaults = defaults.get(tab_name, {})
        assert isinstance(order, list) and len(order) > 0, f"{tab_name}: empty ORDER"
        assert isinstance(tab_defaults, dict), f"{tab_name}: defaults missing dict"
        for key in order:
            assert key in tab_defaults, f"{tab_name}: missing default key '{key}'"

        vals = dict_to_values(tab_name, tab_defaults, tab_defaults)
        assert len(vals) == len(order), f"{tab_name}: values/order length mismatch"
        back = values_to_dict(tab_name, vals)
        for key in order:
            assert key in back, f"{tab_name}: roundtrip missing key '{key}'"


def _check_batch_processor_api() -> None:
    jobs = [BatchJob(input_path="a"), BatchJob(input_path="b")]
    bp = BatchProcessor(telemetry_enabled=False)

    def proc(job: BatchJob) -> bool:
        job.output_path = f"{job.input_path}_out"
        return True

    res = bp.process_batch(jobs=jobs, processor_func=proc, max_concurrent=1)
    assert res.total_files == 2
    assert res.completed_files == 2
    assert all(j.status == "completed" for j in jobs)


def main() -> None:
    _check_universal_preset_caches()
    _check_universal_schema_roundtrip()
    _check_batch_processor_api()
    print("SMOKE: validation passed")


if __name__ == "__main__":
    main()
