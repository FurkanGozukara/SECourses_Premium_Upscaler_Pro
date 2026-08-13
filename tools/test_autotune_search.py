from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from shared.services.autotune_search import (
    ambient_adjusted_min_device_free_gb,
    ambient_adjusted_peak_gb,
    autotune_launch_headroom,
    cached_best_has_headroom,
    frontier_bisect,
    is_reusable_vram_boundary_outcome,
    is_verified_autotune_payload,
    is_vram_boundary_outcome,
    persisted_autotune_status,
    resolution_signatures_compatible,
    resolution_signatures_identical,
    resume_frontier_hints,
    vram_drain_target_gb,
)


def run_frontier(
    outcomes,
    *,
    count: int,
    hint=None,
    growth_step: int = 4,
    early=None,
    reused=None,
    trusted_pass=None,
    known_boundary=None,
):
    calls = []
    early = set(early or ())
    reused = set(reused or ())

    def probe(index: int, require_full: bool):
        if False:
            yield None
        calls.append((index, require_full))
        value = outcomes(index, require_full) if callable(outcomes) else outcomes[index]
        return {
            "outcome": value,
            "early_stopped_pass": bool(index in early and not require_full),
            "reused_outcome": bool(index in reused),
        }

    search = frontier_bisect(
        count,
        probe,
        initial_index=hint,
        max_growth_step=growth_step,
        trusted_pass_index=trusted_pass,
        known_boundary_index=known_boundary,
    )
    try:
        while True:
            next(search)
    except StopIteration as done:
        return done.value, calls


class FrontierSearchTests(unittest.TestCase):
    def test_all_monotone_frontiers_settle_on_the_exact_best(self):
        for count in range(1, 41):
            for best in range(-1, count):
                with self.subTest(count=count, best=best):
                    result, calls = run_frontier(
                        lambda index, _full, limit=best: (
                            "pass" if index <= limit else "boundary_fail"
                        ),
                        count=count,
                        hint=count - 1,
                        growth_step=4,
                    )
                    self.assertEqual(result["best_idx"], best if best >= 0 else None)
                    self.assertEqual(calls[0], (0, False))
                    self.assertEqual(result["frontier_verified"], best >= 0)

    def test_starts_safe_and_uses_bounded_growth_then_bisection(self):
        result, calls = run_frontier(
            lambda index, _full: "pass" if index <= 15 else "boundary_fail",
            count=51,
            hint=40,
            growth_step=4,
        )

        self.assertEqual(result["best_idx"], 15)
        self.assertTrue(result["frontier_verified"])
        self.assertEqual([idx for idx, _ in calls], [0, 4, 8, 12, 16, 14, 15])
        self.assertTrue(all((right - left) <= 4 for left, right in zip([0, 4, 8, 12], [4, 8, 12, 16])))

    def test_partial_campaign_runs_only_the_missing_frontier_tests(self):
        saved = {0: "pass", 4: "boundary_fail"}
        live_calls = []

        def outcome(index, _full):
            if index in saved:
                return saved[index]
            live_calls.append(index)
            return "pass" if index <= 2 else "boundary_fail"

        result, _calls = run_frontier(outcome, count=5, growth_step=4)

        self.assertEqual(result["best_idx"], 2)
        self.assertTrue(result["frontier_verified"])
        self.assertEqual(live_calls, [2, 3])

    def test_saved_pass_and_vram_boundary_only_run_missing_configs(self):
        saved_outcomes = [None] * 10
        saved_outcomes[4] = {
            "passed": True,
            "returncode": 0,
            "telemetry_ok": True,
            "oom": False,
            "probe_cancel_reason": "",
            "estimated_free_gb": 5.0,
        }
        saved_outcomes[8] = {
            "passed": False,
            "returncode": 1,
            "telemetry_ok": True,
            "oom": False,
            "probe_cancel_reason": "threshold_reached",
            "estimated_free_gb": 1.9,
        }
        trusted_pass, known_boundary = resume_frontier_hints(saved_outcomes, 2.0)
        live_calls = []

        def outcome(index, _full):
            if saved_outcomes[index] is not None:
                return "pass" if index == 4 else "boundary_fail"
            live_calls.append(index)
            return "pass" if index <= 6 else "boundary_fail"

        result, calls = run_frontier(
            outcome,
            count=10,
            growth_step=4,
            reused={4, 8},
            trusted_pass=trusted_pass,
            known_boundary=known_boundary,
        )

        self.assertEqual((trusted_pass, known_boundary), (4, 8))
        self.assertEqual(result["best_idx"], 6)
        self.assertTrue(result["frontier_verified"])
        self.assertEqual([index for index, _full in calls], [4, 8, 6, 7])
        self.assertEqual(live_calls, [6, 7])
        self.assertTrue(all(index < 8 for index in live_calls))

    def test_every_partial_monotone_campaign_resumes_to_the_same_frontier(self):
        for count in range(1, 9):
            for best in range(-1, count):
                for saved_mask in range(1 << count):
                    with self.subTest(count=count, best=best, saved_mask=saved_mask):
                        saved_indices = {
                            index for index in range(count) if saved_mask & (1 << index)
                        }
                        saved_outcomes = []
                        for index in range(count):
                            if index not in saved_indices:
                                saved_outcomes.append(None)
                            elif index <= best:
                                saved_outcomes.append(
                                    {
                                        "passed": True,
                                        "returncode": 0,
                                        "telemetry_ok": True,
                                        "oom": False,
                                        "probe_cancel_reason": "",
                                        "estimated_free_gb": 4.0,
                                    }
                                )
                            else:
                                saved_outcomes.append(
                                    {
                                        "passed": False,
                                        "returncode": 1,
                                        "telemetry_ok": True,
                                        "oom": False,
                                        "probe_cancel_reason": "threshold_reached",
                                        "estimated_free_gb": 1.5,
                                    }
                                )

                        trusted_pass, known_boundary = resume_frontier_hints(
                            saved_outcomes,
                            2.0,
                        )
                        live_calls = []

                        def outcome(index, _full, frontier=best):
                            if index not in saved_indices:
                                live_calls.append(index)
                            return "pass" if index <= frontier else "boundary_fail"

                        result, calls = run_frontier(
                            outcome,
                            count=count,
                            growth_step=3,
                            reused=saved_indices,
                            trusted_pass=trusted_pass,
                            known_boundary=known_boundary,
                        )

                        self.assertEqual(result["best_idx"], best if best >= 0 else None)
                        called_indices = [index for index, _full in calls]
                        self.assertEqual(len(called_indices), len(set(called_indices)))
                        self.assertTrue(saved_indices.isdisjoint(live_calls))
                        if known_boundary is not None:
                            self.assertTrue(all(index < known_boundary for index in live_calls))

    def test_resume_hints_only_trust_clean_full_passes(self):
        outcomes = [
            {
                "passed": True,
                "returncode": 1,
                "telemetry_ok": True,
                "oom": False,
                "probe_cancel_reason": "stable_pass",
                "estimated_free_gb": 8.0,
            },
            {
                "passed": False,
                "returncode": 1,
                "telemetry_ok": True,
                "oom": False,
                "probe_cancel_reason": "",
                "estimated_free_gb": 8.0,
            },
            {
                "passed": True,
                "returncode": 0,
                "telemetry_ok": True,
                "oom": False,
                "probe_cancel_reason": "",
                "estimated_free_gb": 4.0,
            },
            {
                "passed": False,
                "returncode": 1,
                "telemetry_ok": False,
                "oom": True,
                "probe_cancel_reason": "",
                "estimated_free_gb": 0.0,
            },
        ]

        self.assertEqual(resume_frontier_hints(outcomes, 2.0), (2, 3))

    def test_saved_hard_failures_do_not_prevent_missing_tests(self):
        live_calls = []

        def outcome(index, _full):
            if index < 3:
                return "hard_fail"
            live_calls.append(index)
            return "pass" if index == 3 else "boundary_fail"

        result, _calls = run_frontier(
            outcome,
            count=5,
            growth_step=1,
            reused={0, 1, 2},
        )

        self.assertEqual(result["best_idx"], 3)
        self.assertEqual(live_calls, [3, 4])
        self.assertTrue(result["frontier_verified"])

    def test_non_vram_failure_is_skipped_without_losing_higher_passes(self):
        def outcome(index, _full):
            if index == 12:
                return "hard_fail"
            return "pass" if index <= 15 else "boundary_fail"

        result, calls = run_frontier(outcome, count=25, growth_step=4)

        self.assertEqual(result["best_idx"], 15)
        self.assertIn(12, result["hard_failed_indices"])
        self.assertFalse(result["indeterminate_above_best"])
        self.assertTrue(result["frontier_verified"])
        self.assertIn((15, False), calls)

    def test_hard_failure_above_best_prevents_cache_verification(self):
        def outcome(index, _full):
            if index == 16:
                return "hard_fail"
            return "pass" if index <= 15 else "boundary_fail"

        result, _calls = run_frontier(outcome, count=25, growth_step=4)

        self.assertEqual(result["best_idx"], 15)
        self.assertTrue(result["indeterminate_above_best"])
        self.assertFalse(result["frontier_verified"])
        self.assertIsNone(result["stopped"])

    def test_each_candidate_is_probed_at_most_once(self):
        result, calls = run_frontier(
            lambda index, _full: "pass" if index <= 3 else "boundary_fail",
            count=6,
            growth_step=2,
            early={3},
        )

        self.assertEqual(result["best_idx"], 3)
        self.assertEqual(len(calls), len({index for index, _full in calls}))
        self.assertFalse(any(require_full for _index, require_full in calls))
        self.assertTrue(result["frontier_verified"])

    def test_seedvr2_failed_campaign_pattern_no_longer_selects_batch_49(self):
        batches = list(range(5, 202, 4))
        old_hard_failures = {121, 201}

        def outcome(index, _full):
            batch = batches[index]
            if batch in old_hard_failures:
                return "hard_fail"
            return "pass" if batch <= 65 else "boundary_fail"

        result, calls = run_frontier(
            outcome,
            count=len(batches),
            hint=len(batches) - 1,
            growth_step=4,
        )

        self.assertEqual(batches[result["best_idx"]], 65)
        self.assertEqual(batches[calls[0][0]], 5)
        self.assertNotIn(201, [batches[index] for index, _full in calls])
        self.assertTrue(result["frontier_verified"])

    def test_safest_candidate_boundary_stops_without_larger_probe(self):
        result, calls = run_frontier(
            lambda _index, _full: "boundary_fail",
            count=20,
            hint=19,
            growth_step=4,
        )

        self.assertIsNone(result["best_idx"])
        self.assertEqual(calls, [(0, False)])
        self.assertFalse(result["frontier_verified"])

    def test_repeated_initial_runtime_failures_abort_the_broken_sweep(self):
        result, calls = run_frontier(
            lambda _index, _full: "hard_fail",
            count=50,
            hint=49,
            growth_step=4,
        )

        self.assertEqual(result["stopped"], "hard_fail")
        self.assertEqual([index for index, _full in calls], [0, 1, 2])
        self.assertIsNone(result["best_idx"])


class AutoTuneCacheTests(unittest.TestCase):
    def test_vram_drain_target_stays_above_a_busy_gpu_baseline(self):
        # A 32GB GPU with 18GB legitimately occupied used to wait for an
        # impossible <=16GB target after every probe.
        self.assertEqual(vram_drain_target_gb(18.0), 19.5)
        self.assertGreater(vram_drain_target_gb(18.0), 18.0)

    def test_launch_guard_requires_reserve_plus_margin(self):
        self.assertEqual(
            autotune_launch_headroom(32.0, 29.5, 2.0),
            (False, 2.5, 3.0),
        )
        self.assertEqual(
            autotune_launch_headroom(32.0, 28.0, 2.0),
            (True, 4.0, 3.0),
        )

    def test_lower_current_ambient_forces_a_new_optimality_scan(self):
        payload = {
            "gpu": {"ambient_used_gb": 8.0},
            "best_config": {
                "measured_peak_vram_used_gb": 24.0,
                "min_device_free_vram_gb": 8.0,
            },
        }
        signature = {"gpu_total_vram_gb": 32.0}

        self.assertFalse(
            cached_best_has_headroom(
                payload,
                signature,
                2.0,
                current_ambient_used_gb=1.0,
            )
        )
        self.assertTrue(
            cached_best_has_headroom(
                payload,
                signature,
                2.0,
                current_ambient_used_gb=8.2,
            )
        )

    def test_lower_reserve_reruns_an_old_threshold_stop(self):
        stopped = {
            "oom": False,
            "telemetry_ok": True,
            "estimated_free_gb": 2.9,
            "probe_cancel_reason": "threshold_reached",
        }
        self.assertFalse(
            is_reusable_vram_boundary_outcome(
                stopped,
                2.0,
                recorded_min_free_vram_gb=3.0,
            )
        )
        self.assertTrue(
            is_reusable_vram_boundary_outcome(
                stopped,
                3.0,
                recorded_min_free_vram_gb=3.0,
            )
        )
        self.assertFalse(
            is_reusable_vram_boundary_outcome(
                stopped,
                2.0,
                recorded_min_free_vram_gb=float("nan"),
            )
        )

    def test_failed_or_unverified_payload_is_never_reusable(self):
        base = {"best_config": {"batch_size": 49}, "finalized": True, "frontier_verified": True}
        self.assertFalse(is_verified_autotune_payload({**base, "status": "failed"}))
        self.assertFalse(
            is_verified_autotune_payload({**base, "status": "completed", "frontier_verified": False})
        )
        self.assertFalse(is_verified_autotune_payload({**base, "status": "threshold_reached"}))
        self.assertTrue(is_verified_autotune_payload({**base, "status": "completed"}))

    def test_verified_frontier_is_persisted_as_a_completed_campaign(self):
        self.assertEqual(
            persisted_autotune_status(
                "threshold_reached",
                finalized=True,
                frontier_verified=True,
            ),
            "completed",
        )
        self.assertEqual(
            persisted_autotune_status(
                "cancelled",
                finalized=True,
                frontier_verified=False,
            ),
            "cancelled",
        )

    def test_equal_pixels_with_different_aspect_ratio_do_not_match(self):
        widescreen = {"target_width": 1920, "target_height": 1080, "target_pixels": 2_073_600}
        square = {"target_width": 1440, "target_height": 1440, "target_pixels": 2_073_600}
        nearby = {"target_width": 1904, "target_height": 1072, "target_pixels": 2_041_088}

        self.assertFalse(resolution_signatures_compatible(square, widescreen))
        self.assertTrue(resolution_signatures_compatible(nearby, widescreen))

    def test_same_output_with_different_model_input_size_does_not_match(self):
        small_preprocess = {
            "target_width": 1920,
            "target_height": 1080,
            "target_pixels": 2_073_600,
            "effective_input_width": 480,
            "effective_input_height": 270,
        }
        large_preprocess = {
            **small_preprocess,
            "effective_input_width": 960,
            "effective_input_height": 540,
        }

        self.assertFalse(resolution_signatures_compatible(small_preprocess, large_preprocess))

    def test_final_cache_reuse_requires_exact_resolved_shapes(self):
        measured = {
            "target_width": 1920,
            "target_height": 1080,
            "effective_input_width": 480,
            "effective_input_height": 270,
        }
        nearby = {**measured, "target_width": 1904, "target_height": 1072}

        self.assertTrue(resolution_signatures_identical(measured, dict(measured)))
        self.assertFalse(resolution_signatures_identical(measured, nearby))
        self.assertTrue(resolution_signatures_compatible(measured, nearby))

    def test_runtime_crash_with_headroom_is_not_a_vram_boundary(self):
        crash = {
            "returncode": 1,
            "telemetry_ok": True,
            "oom": False,
            "probe_cancel_reason": "",
            "estimated_free_gb": 15.0,
        }
        threshold = {**crash, "probe_cancel_reason": "threshold_reached", "estimated_free_gb": 0.5}

        self.assertFalse(is_vram_boundary_outcome(crash, 2.0))
        self.assertTrue(is_vram_boundary_outcome(threshold, 2.0))

    def test_explicit_oom_is_a_boundary_even_if_sampler_missed_telemetry(self):
        oom = {
            "returncode": 1,
            "telemetry_ok": False,
            "oom": True,
            "probe_cancel_reason": "",
            "estimated_free_gb": 0.0,
        }

        self.assertTrue(is_vram_boundary_outcome(oom, 2.0))

    def test_cached_peak_is_rechecked_for_new_reserve_and_ambient_usage(self):
        payload = {
            "best_config": {"measured_peak_vram_used_gb": 27.0},
            "gpu": {"ambient_used_gb": 1.0},
        }
        signature = {"gpu_total_vram_gb": 32.0}

        self.assertTrue(cached_best_has_headroom(payload, signature, 2.0, current_ambient_used_gb=1.0))
        self.assertFalse(cached_best_has_headroom(payload, signature, 6.0, current_ambient_used_gb=1.0))
        self.assertFalse(cached_best_has_headroom(payload, signature, 2.0, current_ambient_used_gb=4.5))
        self.assertEqual(ambient_adjusted_peak_gb(27.0, payload, 4.5), 30.5)

    def test_per_device_headroom_is_adjusted_conservatively(self):
        self.assertEqual(
            ambient_adjusted_min_device_free_gb(3.0, 27.0, 28.5, 3.5),
            1.5,
        )
        payload = {
            "best_config": {
                "measured_peak_vram_used_gb": 27.0,
                "min_device_free_vram_gb": 3.0,
            },
            "gpu": {"ambient_used_gb": 1.0},
        }
        self.assertFalse(
            cached_best_has_headroom(
                payload,
                {"gpu_total_vram_gb": 32.0},
                2.0,
                current_ambient_used_gb=2.5,
            )
        )


class ProcessControlTests(unittest.TestCase):
    def test_forced_probe_stop_reaps_a_silent_process(self):
        from shared.process_control import terminate_process_tree

        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            terminate_process_tree(proc, grace_sec=0.05)
            self.assertIsNotNone(proc.poll())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2.0)


class SparkSignatureTests(unittest.TestCase):
    def test_temporal_context_always_outranks_spatial_improvement(self):
        from shared.services.sparkvsr_autotune import _quality_rank

        shorter_full_frame = _quality_rank({"chunk_len": 73, "tile_height": 0})
        longer_small_tile = _quality_rank({"chunk_len": 74, "tile_height": 256})
        self.assertGreater(longer_small_tile, shorter_full_frame)

    def test_growth_candidates_include_exact_bounded_source_length(self):
        from shared.services.sparkvsr_autotune import _growth_candidate_settings

        self.assertEqual(
            [item["chunk_len"] for item in _growth_candidate_settings(72)],
            [72],
        )
        self.assertEqual(
            [item["chunk_len"] for item in _growth_candidate_settings(100)],
            [73, 81, 89, 97, 100],
        )

    def test_signature_separates_offload_and_reference_memory_lanes(self):
        from shared.services.sparkvsr_autotune import _build_autotune_signature

        settings = {
            "model_name": "SparkVSR-S2",
            "model_path": "models/spark",
            "lora_path": "models/detail.safetensors",
            "group_offload": True,
            "num_blocks_per_group": 4,
            "ref_mode": "sr_image",
            "ref_indices": "0, 20, 40",
        }
        signature = _build_autotune_signature(
            settings,
            target_w=1920,
            target_h=1080,
            effective_in_w=480,
            effective_in_h=270,
            total_vram_gb=32.0,
            min_free_target_gb=2.0,
            allow_full_sequence=False,
            growth_probe_frames=129,
        )

        exact = signature["exact"]
        self.assertTrue(exact["group_offload"])
        self.assertEqual(exact["num_blocks_per_group"], 4)
        self.assertEqual(exact["reference_count"], 3)
        self.assertEqual(exact["lora_path"], "models/detail.safetensors")

    def test_spark_cache_requires_axis_and_full_pass_proof(self):
        from shared.services.sparkvsr_autotune import (
            _build_autotune_signature,
            _find_cached_autotune_log,
        )

        signature = _build_autotune_signature(
            {"model_name": "SparkVSR-S2", "ref_mode": "no_ref"},
            target_w=1920,
            target_h=1080,
            effective_in_w=480,
            effective_in_h=270,
            total_vram_gb=32.0,
            min_free_target_gb=2.0,
            allow_full_sequence=False,
            growth_probe_frames=129,
        )
        payload = {
            "status": "completed",
            "finalized": True,
            "frontier_verified": True,
            "signature": signature,
            "gpu": {"ambient_used_gb": 0.0},
            "best_config": {
                "chunk_len": 65,
                "tile_height": 256,
                "tile_width": 256,
                "measured_peak_vram_used_gb": 20.0,
            },
            "tests": [],
            "search_results": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sparkvsr_autotune_test.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(_find_cached_autotune_log(Path(tmp), signature, 2.0, 0.0))

            payload["tests"] = [
                {
                    "chunk_len": 65,
                    "tile_height": 256,
                    "tile_width": 256,
                    "returncode": 0,
                    "telemetry_ok": True,
                    "passed": True,
                    "probe_cancel_reason": "",
                    "split_stage_validation_ok": True,
                    "reference_count": 0,
                }
            ]
            payload["search_results"] = {
                "temporal": {"frontier_verified": True, "hard_failed_indices": []},
                "spatial": {"frontier_verified": True, "hard_failed_indices": []},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNotNone(_find_cached_autotune_log(Path(tmp), signature, 2.0, 0.0))
            legacy_signature = json.loads(json.dumps(signature))
            legacy_signature["exact"].update(
                global_gpu_device="7",
                gpu_identity="7:some other gpu:9.9",
            )
            legacy_signature["exact_hash"] = "legacy-device-identity"
            payload["signature"] = legacy_signature
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNotNone(_find_cached_autotune_log(Path(tmp), signature, 2.0, 0.0))
            other_capacity = json.loads(json.dumps(signature))
            other_capacity["gpu_total_vram_gb"] = 24.0
            self.assertIsNone(_find_cached_autotune_log(Path(tmp), other_capacity, 2.0, 0.0))
            other_reserve = json.loads(json.dumps(signature))
            other_reserve["exact"]["save_vram_gb"] = 3.0
            other_reserve["exact_hash"] = "different-reserve"
            self.assertIsNone(_find_cached_autotune_log(Path(tmp), other_reserve, 3.0, 0.0))


class FlashStrategyTests(unittest.TestCase):
    def test_autotune_never_probes_below_the_runner_tiled_dit_minimum(self):
        from shared.flashvsr_runner import FLASHVSR_MIN_TILED_DIT_TILE_SIZE
        from shared.services.flashvsr_autotune import AUTOTUNE_FALLBACK_TILES

        self.assertEqual(
            min(AUTOTUNE_FALLBACK_TILES),
            FLASHVSR_MIN_TILED_DIT_TILE_SIZE,
        )

    def test_stream_decode_only_forces_untiled_in_supported_modes(self):
        from shared.services.flashvsr_autotune import _stream_decode_forces_untiled

        self.assertTrue(_stream_decode_forces_untiled({"stream_decode": True, "mode": "tiny"}))
        self.assertTrue(
            _stream_decode_forces_untiled({"stream_decode": True, "mode": "tiny-long"})
        )
        self.assertFalse(_stream_decode_forces_untiled({"stream_decode": True, "mode": "full"}))
        self.assertFalse(_stream_decode_forces_untiled({"stream_decode": False, "mode": "tiny"}))

    def test_flash_cache_requires_a_full_pass_matching_the_best(self):
        from shared.services.flashvsr_autotune import (
            _build_autotune_signature,
            _find_cached_autotune_log,
        )

        signature = _build_autotune_signature(
            {"version": "1.1", "mode": "full", "precision": "bf16"},
            target_w=1920,
            target_h=1080,
            effective_in_w=480,
            effective_in_h=270,
            total_vram_gb=32.0,
            min_free_target_gb=2.0,
        )
        best = {
            "frame_chunk_size": 450,
            "tile_size": 1024,
            "overlap": 48,
            "tiled_dit": False,
            "measured_peak_vram_used_gb": 20.0,
        }
        probe = {
            **best,
            "returncode": 1,
            "telemetry_ok": True,
            "passed": True,
            "probe_cancel_reason": "stable_pass",
            "phase2_gate_ready": True,
            "iter_idx_max": 3,
        }
        payload = {
            "model": "flashvsrplus",
            "status": "completed",
            "finalized": True,
            "frontier_verified": True,
            "signature": signature,
            "gpu": {"ambient_used_gb": 0.0},
            "best_config": best,
            "tests": [probe],
            "search_results": {
                "temporal": {"frontier_verified": True, "hard_failed_indices": []},
                "spatial_fallback": {"frontier_verified": True, "hard_failed_indices": []},
                "spatial_primary": {"frontier_verified": True, "hard_failed_indices": []},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flashvsrplus_autotune_test.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(_find_cached_autotune_log(Path(tmp), signature, 2.0, 0.0))

            payload["tests"][0].update(returncode=0, probe_cancel_reason="")
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNotNone(_find_cached_autotune_log(Path(tmp), signature, 2.0, 0.0))
            legacy_signature = json.loads(json.dumps(signature))
            legacy_signature["exact"].update(
                global_gpu_device="7",
                gpu_identity="7:some other gpu:9.9",
            )
            legacy_signature["exact_hash"] = "legacy-device-identity"
            payload["signature"] = legacy_signature
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNotNone(_find_cached_autotune_log(Path(tmp), signature, 2.0, 0.0))
            other_capacity = json.loads(json.dumps(signature))
            other_capacity["gpu_total_vram_gb"] = 24.0
            self.assertIsNone(_find_cached_autotune_log(Path(tmp), other_capacity, 2.0, 0.0))
            other_reserve = json.loads(json.dumps(signature))
            other_reserve["exact"]["save_vram_gb"] = 3.0
            other_reserve["exact_hash"] = "different-reserve"
            self.assertIsNone(_find_cached_autotune_log(Path(tmp), other_reserve, 3.0, 0.0))


class SeedVaeRecoveryTests(unittest.TestCase):
    def test_blockswap_ceiling_matches_each_seed_architecture(self):
        from shared.services.seedvr2_service import _autotune_max_blocks_for_model

        self.assertEqual(
            _autotune_max_blocks_for_model("seedvr2_ema_3b_fp16.safetensors"),
            32,
        )
        self.assertEqual(
            _autotune_max_blocks_for_model("seedvr2_ema_7b_fp16.safetensors"),
            36,
        )
        self.assertEqual(
            _autotune_max_blocks_for_model(
                "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
            ),
            35,
        )

    def test_phase_specific_tile_recovery_does_not_slow_the_other_vae_phase(self):
        from shared.services.seedvr2_service import _halve_vae_tile_sizes

        settings = {
            "vae_encode_tile_size": 1024,
            "vae_encode_tile_overlap": 128,
            "vae_decode_tile_size": 1024,
            "vae_decode_tile_overlap": 128,
        }
        changed, _notes = _halve_vae_tile_sizes(settings, encode=True, decode=False)

        self.assertTrue(changed)
        self.assertEqual(settings["vae_encode_tile_size"], 512)
        self.assertEqual(settings["vae_encode_tile_overlap"], 64)
        self.assertEqual(settings["vae_decode_tile_size"], 1024)
        self.assertEqual(settings["vae_decode_tile_overlap"], 128)

    def test_seed_cache_requires_a_full_pass_matching_the_best(self):
        from shared.services.seedvr2_service import (
            AUTOTUNE_TARGET_FRAMES,
            _build_autotune_signature,
            _find_cached_autotune_log,
        )

        settings = {
            "dit_model": "seedvr2_ema_7b_sharp_fp16.safetensors",
            "vae_encode_tiled": True,
            "vae_decode_tiled": True,
            "vae_encode_tile_size": 1024,
            "vae_decode_tile_size": 1024,
        }
        signature = _build_autotune_signature(
            settings,
            target_w=1904,
            target_h=1072,
            effective_in_w=480,
            effective_in_h=270,
            total_vram_gb=32.0,
            min_free_target_gb=2.0,
        )
        best = {
            "batch_size": AUTOTUNE_TARGET_FRAMES,
            "blocks_to_swap": 0,
            "vae_encode_tiled": True,
            "vae_decode_tiled": True,
            "vae_encode_tile_size": 1024,
            "vae_decode_tile_size": 1024,
            "measured_peak_vram_used_gb": 20.0,
        }
        payload = {
            "model": "seedvr2",
            "status": "completed",
            "finalized": True,
            "frontier_verified": True,
            "signature": signature,
            "gpu": {"ambient_used_gb": 0.0},
            "best_config": best,
            "tests": [],
            "search_results": {
                "batch": {"frontier_verified": True, "hard_failed_indices": []},
                "blocks_to_swap": {"frontier_verified": True, "hard_failed_indices": []},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seedvr2_autotune_test.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(_find_cached_autotune_log(Path(tmp), signature, 2.0, 0.0))

            payload["tests"] = [
                {
                    **best,
                    "returncode": 0,
                    "telemetry_ok": True,
                    "oom": False,
                    "passed": True,
                    "probe_cancel_reason": "",
                }
            ]
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNotNone(_find_cached_autotune_log(Path(tmp), signature, 2.0, 0.0))
            legacy_signature = json.loads(json.dumps(signature))
            legacy_signature["exact"].update(
                global_gpu_device="7",
                gpu_identity="7:some other gpu:9.9",
            )
            legacy_signature["exact_hash"] = "legacy-device-identity"
            payload["signature"] = legacy_signature
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNotNone(_find_cached_autotune_log(Path(tmp), signature, 2.0, 0.0))
            other_capacity = json.loads(json.dumps(signature))
            other_capacity["gpu_total_vram_gb"] = 24.0
            self.assertIsNone(_find_cached_autotune_log(Path(tmp), other_capacity, 2.0, 0.0))
            other_reserve = json.loads(json.dumps(signature))
            other_reserve["exact"]["save_vram_gb"] = 3.0
            other_reserve["exact_hash"] = "different-reserve"
            self.assertIsNone(_find_cached_autotune_log(Path(tmp), other_reserve, 3.0, 0.0))


if __name__ == "__main__":
    unittest.main()
