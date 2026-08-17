#!/usr/bin/env python3
"""Report PBS GPU queue pressure and rough wait bounds for one job name."""

from __future__ import annotations

import argparse
import getpass
import re
import subprocess
from collections import Counter
from datetime import datetime, timedelta


def run_command(command, timeout=60, required=True):
    try:
        return subprocess.run(
            command,
            check=required,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        if required:
            raise SystemExit(f"Command failed: {' '.join(command)}\n{exc}") from exc
        return ""


def parse_records(text, header_prefix):
    records = []
    current = None
    key = None
    property_indent = None
    for line in text.splitlines():
        if line.startswith(header_prefix):
            if current:
                records.append(current)
            current = {"id": line.split(":", 1)[1].strip()}
            key = None
            property_indent = None
            continue
        if current is None:
            continue
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if key and stripped and property_indent is not None and indent > property_indent:
            current[key] += " " + stripped
            continue
        match = re.match(r"^([\w.]+)\s*=\s*(.*)$", stripped)
        if match:
            key, value = match.groups()
            current[key] = value.strip()
            if property_indent is None:
                property_indent = indent
        elif key and line[:1].isspace() and line.strip():
            current[key] += " " + line.strip()
    if current:
        records.append(current)
    return records


def parse_nodes(text):
    nodes = []
    current = None
    key = None
    for line in text.splitlines():
        if line and not line[0].isspace():
            if current:
                nodes.append(current)
            current = {"id": line.strip()}
            key = None
            continue
        if current is None:
            continue
        match = re.match(r"^\s+([\w.]+)\s*=\s*(.*)$", line)
        if match:
            key, value = match.groups()
            current[key] = value.strip()
        elif key and line[:1].isspace() and line.strip():
            current[key] += " " + line.strip()
    if current:
        nodes.append(current)
    return nodes


def parse_datetime(value):
    for fmt in ("%a %b %d %H:%M:%S %Y", "%c"):
        try:
            return datetime.strptime(value, fmt)
        except (TypeError, ValueError):
            pass
    return datetime.min


def parse_duration(value):
    try:
        fields = [int(part) for part in str(value).split(":")]
        if len(fields) != 3:
            return 0
        hours, minutes, seconds = fields
        return hours * 3600 + minutes * 60 + seconds
    except (TypeError, ValueError):
        return 0


def format_duration(seconds):
    if seconds is None:
        return "unavailable"
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes = seconds // 60
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def integer_field(record, key):
    match = re.search(r"\d+", record.get(key, ""))
    return int(match.group()) if match else 0


def gpu_count(job):
    selected = sum(
        int(value)
        for value in re.findall(r"ngpus=(\d+)", job.get("Resource_List.select", ""))
    )
    return selected or integer_field(job, "Resource_List.ngpus")


def gpu_type(job):
    match = re.search(
        r"gpu_type=([^:+\s]+)",
        job.get("Resource_List.select", ""),
    )
    return match.group(1) if match else job.get("Resource_List.gpu_type", "")


def owner(job):
    return job.get("Job_Owner", "unknown").split("@")[0]


def queue_simulation(slot_ready, queued_jobs, target_gpus):
    if not slot_ready or target_gpus > len(slot_ready):
        return None
    slots = sorted(slot_ready)
    for job in queued_jobs:
        requested = gpu_count(job)
        walltime = parse_duration(job.get("Resource_List.walltime", ""))
        if requested <= 0 or requested > len(slots) or walltime <= 0:
            continue
        selected = sorted(range(len(slots)), key=slots.__getitem__)[:requested]
        start = max(slots[index] for index in selected)
        finish = start + walltime
        for index in selected:
            slots[index] = finish
    selected = sorted(slots)[:target_gpus]
    return max(selected)


def eligible_node_capacity(nodes, queue, requested_gpu_type):
    total = 0
    assigned = 0
    count = 0
    for node in nodes:
        node_type = node.get("resources_available.gpu_type", "")
        qlist = node.get(
            "resources_available.Qlist",
            node.get("resources_available.qlist", node.get("queue", "")),
        )
        state = node.get("state", "").lower()
        excluded = {"offline", "down", "stale", "unknown", "provisioning"}
        if requested_gpu_type and requested_gpu_type not in node_type:
            continue
        if qlist and queue not in re.split(r"[,\s]+", qlist):
            continue
        if any(flag in state.split(",") for flag in excluded):
            continue
        available = integer_field(node, "resources_available.ngpus")
        if available <= 0:
            continue
        total += available
        assigned += integer_field(node, "resources_assigned.ngpus")
        count += 1
    return count, total, min(assigned, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--user", default=getpass.getuser())
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    jobs = parse_records(
        run_command(["qstat", "-f"], timeout=args.timeout),
        "Job Id:",
    )
    matches = [
        job
        for job in jobs
        if job.get("Job_Name") == args.job_name and owner(job) == args.user
    ]
    if not matches:
        raise SystemExit(
            f"No active or queued job named {args.job_name!r} for {args.user}."
        )
    target = max(matches, key=lambda job: parse_datetime(job.get("qtime")))
    queue = target.get("queue", "")
    requested_type = gpu_type(target)
    target_gpus = gpu_count(target)

    compatible = [
        job
        for job in jobs
        if job.get("queue") == queue
        and gpu_count(job) > 0
        and gpu_type(job) == requested_type
    ]
    queued = sorted(
        [job for job in compatible if job.get("job_state") == "Q"],
        key=lambda job: parse_datetime(job.get("qtime")),
    )
    rank = next(
        (index for index, job in enumerate(queued) if job["id"] == target["id"]),
        None,
    )
    ahead = queued[:rank] if rank is not None else []
    running = [job for job in compatible if job.get("job_state") == "R"]

    release_slots = []
    for job in running:
        remaining = max(
            0,
            parse_duration(job.get("Resource_List.walltime", ""))
            - parse_duration(job.get("resources_used.walltime", "")),
        )
        release_slots.extend([remaining] * gpu_count(job))

    node_text = run_command(["pbsnodes", "-av"], timeout=args.timeout, required=False)
    node_count, capacity, assigned = eligible_node_capacity(
        parse_nodes(node_text),
        queue,
        requested_type,
    )
    if capacity <= 0:
        capacity = max(len(release_slots), target_gpus)
        assigned = min(len(release_slots), capacity)

    slot_ready = list(release_slots)
    missing_assigned = max(0, assigned - len(slot_ready))
    conservative_unknown = max(slot_ready, default=72 * 3600)
    slot_ready.extend([conservative_unknown] * missing_assigned)
    slot_ready.extend([0] * max(0, capacity - len(slot_ready)))
    if len(slot_ready) > capacity:
        capacity = len(slot_ready)

    capacity_only = queue_simulation(slot_ready, [], target_gpus)
    chronological = queue_simulation(slot_ready, ahead, target_gpus)
    workload_hours = sum(
        gpu_count(job) * parse_duration(job.get("Resource_List.walltime", "")) / 3600
        for job in ahead
    )

    running_gpus_by_owner = Counter()
    running_gpu_hours_by_owner = Counter()
    for job in running:
        job_owner = owner(job)
        job_gpus = gpu_count(job)
        remaining = max(
            0,
            parse_duration(job.get("Resource_List.walltime", ""))
            - parse_duration(job.get("resources_used.walltime", "")),
        )
        running_gpus_by_owner[job_owner] += job_gpus
        running_gpu_hours_by_owner[job_owner] += job_gpus * remaining / 3600

    owner_rows = []
    for job_owner in sorted(set(owner(job) for job in ahead)):
        owner_jobs = [job for job in ahead if owner(job) == job_owner]
        owner_rows.append(
            (
                job_owner,
                len(owner_jobs),
                sum(gpu_count(job) for job in owner_jobs),
                sum(
                    gpu_count(job)
                    * parse_duration(job.get("Resource_List.walltime", ""))
                    / 3600
                    for job in owner_jobs
                ),
                running_gpus_by_owner[job_owner],
                running_gpu_hours_by_owner[job_owner],
                min(ahead.index(job) + 1 for job in owner_jobs),
            )
        )
    owner_rows.sort(key=lambda row: (-row[3], row[6], row[0]))

    target_owner = owner(target)
    own_ahead = [job for job in ahead if owner(job) == target_owner]

    now = datetime.now()
    print("=" * 88)
    print(f"Target: {target['id']}  name={target.get('Job_Name')}  state={target.get('job_state')}")
    print(
        f"Request: {target_gpus} x {requested_type or 'unspecified GPU'}, "
        f"walltime={target.get('Resource_List.walltime', '?')}, queue={queue}"
    )
    print(f"Queued at: {target.get('qtime', '?')}")
    print(f"PBS estimated start: {target.get('estimated.start_time', 'not provided')}")
    print(f"PBS comment: {target.get('comment', 'none')}")
    print()
    print(
        "Chronological compatible-GPU position: "
        f"{rank + 1 if rank is not None else '?'} / {len(queued)}"
    )
    print(f"Jobs before target: {len(ahead)}")
    print(f"Requested GPUs before target: {sum(gpu_count(job) for job in ahead)}")
    print(f"Requested GPU-hours before target: {workload_hours:,.1f}")
    print(f"Jobs ahead by owner: {dict(Counter(owner(job) for job in ahead).most_common(10))}")
    print(f"Distinct owners before target: {len(owner_rows)}")
    print(
        f"Your own jobs before target: {len(own_ahead)}; "
        f"currently running compatible GPUs: {running_gpus_by_owner[target_owner]}"
    )
    print()
    print("All owners represented before target:")
    print(
        f"  {'OWNER':<12} {'QJOBS':>6} {'QGPUS':>6} {'Q_GPUH':>10} "
        f"{'RUN_GPU':>8} {'RUN_REMAIN_GPUH':>15} {'FIRST_POS':>10}"
    )
    for row in owner_rows:
        marker = "*" if row[0] == target_owner else " "
        print(
            f"{marker} {row[0]:<12} {row[1]:>6} {row[2]:>6} {row[3]:>10.1f} "
            f"{row[4]:>8} {row[5]:>15.1f} {row[6]:>10}"
        )
    print()
    print("Last 10 jobs immediately before target:")
    for job in ahead[-10:]:
        print(
            f"  {job['id'].split('.')[0]:>8}  {owner(job):<10} "
            f"gpu={gpu_count(job):<2} wall={job.get('Resource_List.walltime', '?'):<9} "
            f"{job.get('Job_Name', '?')}"
        )
    print()
    if node_count:
        print(
            f"Eligible active nodes: {node_count}; GPUs: {capacity}; "
            f"assigned: {assigned}; apparent free: {max(0, capacity - assigned)}"
        )
    else:
        print(f"Eligible capacity inferred from running jobs: {capacity} GPU(s)")
    print("GPU release windows if running jobs use their full walltime:")
    for hours in (1, 2, 4, 5, 8, 12, 24, 48, 72):
        released = sum(1 for seconds in release_slots if seconds <= hours * 3600)
        print(f"  within {hours:>2}h: {released:>3} GPU(s)")
    if release_slots:
        release_minutes = Counter(round(seconds / 60) for seconds in release_slots)
        cumulative = 0
        print("Next exact release milestones (full-walltime assumption):")
        for minutes, count in sorted(release_minutes.items())[:8]:
            cumulative += count
            when = now + timedelta(minutes=minutes)
            print(
                f"  +{format_duration(minutes * 60):>9}: {count:>2} GPU(s), "
                f"cumulative={cumulative:>2}, around {when:%a %d %b %H:%M}"
            )
    print()
    print("Approximate wait estimates:")
    print(
        f"  Optimistic capacity opening: {format_duration(capacity_only)}"
        + (
            f" (around {(now + timedelta(seconds=capacity_only)):%a %d %b %H:%M})"
            if capacity_only is not None
            else ""
        )
    )
    print(
        "  FIFO/full-walltime pressure model: "
        f"{format_duration(chronological)}"
        + (
            f" (around {(now + timedelta(seconds=chronological)):%a %d %b %H:%M})"
            if chronological is not None
            else ""
        )
    )
    print(
        "  These are planning estimates, not strict bounds. Fair-share, backfill, "
        "same-node placement, reservations, early job exits and offline nodes can "
        "move the real start earlier or later; chronological position is not strict FIFO."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
