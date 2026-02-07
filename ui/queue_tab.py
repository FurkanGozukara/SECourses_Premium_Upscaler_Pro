"""
Queue tab UI for monitoring and managing waiting processing jobs.
"""

from __future__ import annotations

import time
from typing import List

import gradio as gr

from shared.processing_queue import get_processing_queue_manager


def _render_waiting_table(waiting_items: List[dict]) -> str:
    if not waiting_items:
        return "No waiting jobs."
    lines = [
        "| Job ID | Tab | Action | Waiting | Submitted |",
        "|---|---|---|---:|---|",
    ]
    for item in waiting_items:
        lines.append(
            f"| `{item['job_id']}` | {item['tab_name']} | {item['action_name']} | "
            f"{item['wait_seconds_text']} | {item['submitted_at_text']} |"
        )
    return "\n".join(lines)


def queue_tab(queue_tab_component) -> None:
    """
    Render queue monitor tab and wire live refresh + delete actions.

    Args:
        queue_tab_component: The gr.Tab instance so we can update its label.
    """
    queue_manager = get_processing_queue_manager()

    gr.Markdown("### Processing Queue")
    gr.Markdown("Monitor waiting jobs and remove queued items before they start.")

    queue_summary = gr.Markdown("Waiting jobs: **0**")
    active_job = gr.Markdown("Active job: **None**")
    waiting_jobs = gr.Markdown("No waiting jobs.")

    waiting_selector = gr.Dropdown(
        label="Waiting jobs",
        choices=[],
        value=[],
        multiselect=True,
        info="Select waiting jobs to remove from queue.",
    )

    with gr.Row():
        refresh_btn = gr.Button("Refresh", variant="secondary")
        delete_btn = gr.Button("Delete Selected", variant="stop")
        clear_btn = gr.Button("Clear All Waiting", variant="stop")

    action_status = gr.Markdown("")
    queue_timer = gr.Timer(value=2.0, active=True)
    queue_sync_signature = gr.State(value="")

    def refresh_queue(selected_ids, previous_signature: str = ""):
        selected_ids = list(selected_ids or [])
        snapshot = queue_manager.snapshot()
        waiting_items = list(snapshot.get("waiting", []))
        waiting_count = int(snapshot.get("waiting_count", 0))
        active = snapshot.get("active")

        if active:
            active_text = (
                f"Active job: **`{active['job_id']}`** | {active['tab_name']} | "
                f"{active['action_name']} | running {active['wait_seconds_text']}"
            )
        else:
            active_text = "Active job: **None**"

        summary_text = f"Waiting jobs: **{waiting_count}**"
        waiting_table = _render_waiting_table(waiting_items)

        choices = [
            (
                f"{item['job_id']} | {item['tab_name']} | {item['action_name']} | {item['wait_seconds_text']}",
                item["job_id"],
            )
            for item in waiting_items
        ]
        waiting_ids = {item["job_id"] for item in waiting_items}
        valid_selected = [job_id for job_id in selected_ids if job_id in waiting_ids]
        time_bucket = int(time.time() // 10) if (waiting_count > 0 or active) else 0
        signature = "|".join(
            [
                str(waiting_count),
                str(active.get("job_id") if isinstance(active, dict) else ""),
                ",".join(sorted(waiting_ids)),
                ",".join(valid_selected),
                str(time_bucket),
            ]
        )
        if signature == str(previous_signature or ""):
            return gr.skip()

        return (
            gr.update(value=summary_text),
            gr.update(value=active_text),
            gr.update(value=waiting_table),
            gr.update(choices=choices, value=valid_selected),
            gr.update(label=f"Queue ({waiting_count})"),
            signature,
        )

    def delete_selected(selected_ids):
        selected_ids = list(selected_ids or [])
        if not selected_ids:
            return gr.update(value="No waiting jobs selected.")
        removed = queue_manager.cancel_waiting(selected_ids)
        if not removed:
            return gr.update(value="No matching waiting jobs found.")
        removed_text = ", ".join(f"`{job_id}`" for job_id in removed)
        return gr.update(value=f"Removed {len(removed)} job(s): {removed_text}")

    def clear_waiting():
        removed = queue_manager.cancel_all_waiting()
        if not removed:
            return gr.update(value="Queue is already empty.")
        return gr.update(value=f"Cleared {len(removed)} waiting job(s).")

    refresh_outputs = [queue_summary, active_job, waiting_jobs, waiting_selector, queue_tab_component, queue_sync_signature]

    refresh_btn.click(
        fn=refresh_queue,
        inputs=[waiting_selector, queue_sync_signature],
        outputs=refresh_outputs,
        queue=False,
        show_progress="hidden",
    )
    queue_tab_component.select(
        fn=refresh_queue,
        inputs=[waiting_selector, queue_sync_signature],
        outputs=refresh_outputs,
        queue=False,
        show_progress="hidden",
    )
    queue_timer.tick(
        fn=refresh_queue,
        inputs=[waiting_selector, queue_sync_signature],
        outputs=refresh_outputs,
        queue=False,
        show_progress="hidden",
    )

    delete_btn.click(
        fn=delete_selected,
        inputs=[waiting_selector],
        outputs=[action_status],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=refresh_queue,
        inputs=[waiting_selector, queue_sync_signature],
        outputs=refresh_outputs,
        queue=False,
        show_progress="hidden",
    )

    clear_btn.click(
        fn=clear_waiting,
        outputs=[action_status],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=refresh_queue,
        inputs=[waiting_selector, queue_sync_signature],
        outputs=refresh_outputs,
        queue=False,
        show_progress="hidden",
    )
