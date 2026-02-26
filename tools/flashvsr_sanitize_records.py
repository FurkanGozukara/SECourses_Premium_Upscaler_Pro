"""
Sanitize FlashVSR sweep records so strict-tiling calibration rows stay clean.

Rules:
- Detect runs where runtime OOM recovery auto-enabled tiled modes.
- Mark such rows as invalid for calibration by forcing:
  - effective_success=False
  - raw_success=False
  - success=False
  - profile_partial=False
  - oom_recovery_override=True
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

OOM_RECOVERY_HINTS = (
    "auto-enabling tiled vae to prevent oom",
    "auto-enabling tiled dit to prevent oom",
)


def _to_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _contains_oom_recovery_override(text: str) -> bool:
    hay = str(text or "").lower()
    return any(h in hay for h in OOM_RECOVERY_HINTS)


def _parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sanitize FlashVSR VRAM records for strict tiling calibration")
    parser.add_argument(
        "--records-csv",
        type=str,
        default=str(base_dir / "outputs" / "flashvsr_vram_sweeps" / "flashvsr_vram_records.csv"),
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    path = Path(args.records_csv).resolve()
    if not path.exists():
        print(f"ERROR: missing records CSV: {path}")
        return 2

    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        header = list(reader.fieldnames or [])

    if not header:
        print(f"ERROR: invalid CSV header: {path}")
        return 2

    if "oom_recovery_override" not in header:
        header.append("oom_recovery_override")

    scanned_logs = 0
    override_rows = 0
    mutated_rows = 0
    missing_logs = 0

    new_rows: List[Dict[str, str]] = []
    for row in rows:
        safe_row: Dict[str, str] = dict(row or {})
        failure_reason = str(safe_row.get("failure_reason") or "")

        override = _to_bool(safe_row.get("oom_recovery_override")) or (
            "oom_recovery_override" in failure_reason.strip().lower()
        )

        if not override:
            log_path = str(safe_row.get("log_file") or "").strip()
            if log_path:
                p = Path(log_path)
                if p.exists() and p.is_file():
                    scanned_logs += 1
                    try:
                        txt = p.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        txt = ""
                    override = _contains_oom_recovery_override(txt)
                else:
                    missing_logs += 1

        if override:
            override_rows += 1
            before = (
                safe_row.get("success", ""),
                safe_row.get("raw_success", ""),
                safe_row.get("effective_success", ""),
                safe_row.get("profile_partial", ""),
                safe_row.get("failure_reason", ""),
                safe_row.get("oom_recovery_override", ""),
            )

            safe_row["success"] = "False"
            safe_row["raw_success"] = "False"
            safe_row["effective_success"] = "False"
            safe_row["profile_partial"] = "False"
            safe_row["oom_recovery_override"] = "True"
            if str(safe_row.get("failure_reason") or "").strip().lower() in {"", "ok", "profile_timeout_ok"}:
                safe_row["failure_reason"] = "oom_recovery_override"

            after = (
                safe_row.get("success", ""),
                safe_row.get("raw_success", ""),
                safe_row.get("effective_success", ""),
                safe_row.get("profile_partial", ""),
                safe_row.get("failure_reason", ""),
                safe_row.get("oom_recovery_override", ""),
            )
            if before != after:
                mutated_rows += 1
        else:
            safe_row["oom_recovery_override"] = "False"

        new_rows.append(safe_row)

    print(f"Rows: {len(rows)}")
    print(f"Logs scanned: {scanned_logs}")
    print(f"Rows with OOM recovery override: {override_rows}")
    print(f"Rows mutated: {mutated_rows}")
    if missing_logs > 0:
        print(f"Rows with missing log files: {missing_logs}")

    if args.dry_run:
        print("Dry run only. CSV not modified.")
        return 0

    backup = path.with_suffix(path.suffix + f".sanitize.{time.strftime('%Y%m%d_%H%M%S')}.bak")
    try:
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backup: {backup}")
    except Exception:
        print("WARN: backup creation failed; continuing with in-place write")

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        for row in new_rows:
            writer.writerow({k: row.get(k, "") for k in header})
    tmp.replace(path)
    print(f"Sanitized CSV written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

