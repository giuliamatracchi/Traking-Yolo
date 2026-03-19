#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


STAGES = ["fusion", "unicycle", "ackermann"]


def log(msg: str) -> None:
    print(f"[run_all_experiments] {msg}", flush=True)


def bash_command(workspace: str, command: str) -> list[str]:
    setup_cmd = f"source {shlex.quote(os.path.join(workspace, 'install', 'setup.bash'))}"
    full_cmd = f"{setup_cmd} && {command}"
    return ["bash", "-lc", full_cmd]


def start_process(
    workspace: str,
    command: str,
    log_file: Path | None = None,
) -> subprocess.Popen:
    stdout_target = None
    stderr_target = None

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_file, "w", encoding="utf-8")
        stdout_target = f
        stderr_target = subprocess.STDOUT
    else:
        f = None

    proc = subprocess.Popen(
        bash_command(workspace, command),
        stdout=stdout_target,
        stderr=stderr_target,
        preexec_fn=os.setsid,
        text=True,
    )
    proc._log_handle = f  # type: ignore[attr-defined]
    return proc


def stop_process(proc: subprocess.Popen | None, name: str, timeout: float = 10.0) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        close_log_handle(proc)
        return

    log(f"Stopping {name} with SIGINT...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        close_log_handle(proc)
        return

    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            close_log_handle(proc)
            return
        time.sleep(0.2)

    log(f"{name} did not stop after SIGINT, sending SIGTERM...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        close_log_handle(proc)
        return

    t1 = time.time()
    while time.time() - t1 < 5.0:
        if proc.poll() is not None:
            close_log_handle(proc)
            return
        time.sleep(0.2)

    log(f"{name} still alive, sending SIGKILL...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass

    close_log_handle(proc)


def close_log_handle(proc: subprocess.Popen) -> None:
    handle = getattr(proc, "_log_handle", None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass


def wait_or_fail(proc: subprocess.Popen, name: str, warmup_sec: float) -> None:
    time.sleep(warmup_sec)
    if proc.poll() is not None:
        raise RuntimeError(f"{name} terminated too early. Check its log.")


def stage_output_dir(output_root: Path, scenario_name: str, stage: str) -> Path:
    return output_root / f"{scenario_name}_{stage}"


def metrics_files(base_dir: Path, stage: str) -> tuple[Path, Path]:
    return (
        base_dir / f"{stage}_metrics_detail.csv",
        base_dir / f"{stage}_metrics_summary.csv",
    )


def run_stage(
    workspace: str,
    scenario_name: str,
    stage: str,
    duration_sec: float,
    pipeline_warmup_sec: float,
    metrics_warmup_sec: float,
    output_root: Path,
    target_track_id: int,
) -> None:
    stage_dir = stage_output_dir(output_root, scenario_name, stage)
    stage_dir.mkdir(parents=True, exist_ok=True)

    pipeline_log = stage_dir / f"{stage}_pipeline.log"
    metrics_log = stage_dir / f"{stage}_metrics.log"

    pipeline_proc = None
    metrics_proc = None

    try:
        log(f"=== Stage: {stage} ===")
        log(f"Output dir: {stage_dir}")

        pipeline_cmd = f"ros2 launch experiments experiment_pipeline.launch.py stage:={stage}"
        pipeline_proc = start_process(workspace, pipeline_cmd, pipeline_log)
        log(f"Pipeline started for stage={stage}")
        wait_or_fail(pipeline_proc, f"{stage} pipeline", pipeline_warmup_sec)

        metrics_cmd = (
            "ros2 run trajectory_metrics trajectory_metrics_node --ros-args "
            f"-p stage:={stage} "
            f"-p output_dir:={shlex.quote(str(stage_dir))} "
            f"-p target_track_id:={target_track_id}"
        )
        metrics_proc = start_process(workspace, metrics_cmd, metrics_log)
        log(f"Metrics node started for stage={stage}")
        wait_or_fail(metrics_proc, f"{stage} metrics node", metrics_warmup_sec)

        log(f"Collecting data for {duration_sec:.1f} s...")
        time.sleep(duration_sec)

    finally:
        stop_process(metrics_proc, f"{stage} metrics node")
        time.sleep(1.0)
        stop_process(pipeline_proc, f"{stage} pipeline")
        time.sleep(2.0)

    detail_csv, summary_csv = metrics_files(stage_dir, stage)

    if not detail_csv.exists():
        raise RuntimeError(f"Missing detail CSV for stage={stage}: {detail_csv}")
    if not summary_csv.exists():
        raise RuntimeError(f"Missing summary CSV for stage={stage}: {summary_csv}")

    log(f"Stage {stage} completed successfully.")


def run_compare(
    workspace: str,
    scenario_name: str,
    output_root: Path,
    compare_out_dir: Path,
) -> None:
    compare_script = Path(workspace) / "src" / "trajectory_metrics" / "tools" / "compare_metrics.py"
    if not compare_script.exists():
        raise RuntimeError(f"compare_metrics.py not found: {compare_script}")

    fusion_dir = stage_output_dir(output_root, scenario_name, "fusion")
    unicycle_dir = stage_output_dir(output_root, scenario_name, "unicycle")
    ackermann_dir = stage_output_dir(output_root, scenario_name, "ackermann")

    cmd = (
        f"python3 {shlex.quote(str(compare_script))} "
        f"--fusion-detail {shlex.quote(str(fusion_dir / 'fusion_metrics_detail.csv'))} "
        f"--fusion-summary {shlex.quote(str(fusion_dir / 'fusion_metrics_summary.csv'))} "
        f"--unicycle-detail {shlex.quote(str(unicycle_dir / 'unicycle_metrics_detail.csv'))} "
        f"--unicycle-summary {shlex.quote(str(unicycle_dir / 'unicycle_metrics_summary.csv'))} "
        f"--ackermann-detail {shlex.quote(str(ackermann_dir / 'ackermann_metrics_detail.csv'))} "
        f"--ackermann-summary {shlex.quote(str(ackermann_dir / 'ackermann_metrics_summary.csv'))} "
        f"--output-dir {shlex.quote(str(compare_out_dir))}"
    )

    log("Running final comparison...")
    subprocess.run(bash_command(workspace, cmd), check=True)
    log(f"Comparison completed: {compare_out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fusion, unicycle, ackermann sequentially, save metrics, then compare them."
    )
    parser.add_argument(
        "--workspace",
        default=os.path.expanduser("~/ros2_humble"),
        help="ROS2 workspace root",
    )
    parser.add_argument(
        "--scenario-name",
        required=True,
        help="Scenario label, e.g. scenario_01",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        required=True,
        help="How long to record metrics for each stage",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory for outputs. Default: <workspace>/metrics_output",
    )
    parser.add_argument(
        "--target-track-id",
        type=int,
        default=-1,
        help="Fixed target track id. Use -1 for auto selection",
    )
    parser.add_argument(
        "--pipeline-warmup-sec",
        type=float,
        default=8.0,
        help="Warmup before starting metrics node",
    )
    parser.add_argument(
        "--metrics-warmup-sec",
        type=float,
        default=3.0,
        help="Warmup after starting metrics node",
    )

    args = parser.parse_args()

    workspace = os.path.abspath(os.path.expanduser(args.workspace))
    output_root = (
        Path(os.path.abspath(os.path.expanduser(args.output_root)))
        if args.output_root is not None
        else Path(workspace) / "metrics_output"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    compare_out_dir = output_root / f"{args.scenario_name}_compare"
    compare_out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(workspace, "install", "setup.bash").exists():
        raise RuntimeError(
            f"Workspace setup not found: {Path(workspace, 'install', 'setup.bash')}. "
            "Build the workspace first."
        )

    try:
        for stage in STAGES:
            run_stage(
                workspace=workspace,
                scenario_name=args.scenario_name,
                stage=stage,
                duration_sec=args.duration_sec,
                pipeline_warmup_sec=args.pipeline_warmup_sec,
                metrics_warmup_sec=args.metrics_warmup_sec,
                output_root=output_root,
                target_track_id=args.target_track_id,
            )

        run_compare(
            workspace=workspace,
            scenario_name=args.scenario_name,
            output_root=output_root,
            compare_out_dir=compare_out_dir,
        )

        log("All stages completed successfully.")
        log(f"Final outputs in: {compare_out_dir}")
        return 0

    except KeyboardInterrupt:
        log("Interrupted by user.")
        return 130
    except Exception as e:
        log(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
