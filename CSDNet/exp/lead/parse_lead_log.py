#!/usr/bin/env python
import argparse
import csv
import os
import re
from collections import defaultdict


TASK_RE = re.compile(
    r"Task start:\s+target=(?P<target>\S+)\s+start_id=(?P<start_id>\d+)\s+"
    r"sim_thr=(?P<sim_thr>[0-9.]+)\s+seed=(?P<seed>\d+)"
)


def new_task(match):
    return {
        "target": match.group("target"),
        "start_id": int(match.group("start_id")),
        "sim_thr": float(match.group("sim_thr")),
        "seed": int(match.group("seed")),
        "generated": "",
        "planned": "",
        "uniqueness_actual": "",
        "unique_over_planned": "",
        "success": False,
        "top_ds": "",
        "top_mol": "",
    }


def parse_log(path):
    rows = []
    current = None

    with open(path, errors="replace") as f:
        for raw in f:
            line = raw.strip()
            match = TASK_RE.search(line)
            if match:
                if current is not None:
                    rows.append(current)
                current = new_task(match)
                continue

            if current is None:
                continue

            if line.startswith("Generated:"):
                val = line.split(":", 1)[1].strip()
                if "/" in val:
                    generated, planned = val.split("/", 1)
                    current["generated"] = generated.strip()
                    current["planned"] = planned.strip()
                else:
                    current["generated"] = val
                continue

            if line.startswith("Uniqueness(actual):"):
                current["uniqueness_actual"] = line.split(":", 1)[1].strip()
                continue

            if line.startswith("Unique/planned:"):
                current["unique_over_planned"] = line.split(":", 1)[1].strip()
                continue

            if line.startswith("Uniqueness:") and not current["uniqueness_actual"]:
                # Backward-compatible parser for older eval output.
                current["unique_over_planned"] = line.split(":", 1)[1].strip()
                continue

            if line.startswith("Lead optimization failed"):
                current["success"] = False
                continue

            if line.startswith("Top DS:"):
                current["success"] = True
                current["top_ds"] = line.split(":", 1)[1].strip()
                continue

            if line.startswith("Top mol:"):
                current["top_mol"] = line.split(":", 1)[1].strip()
                continue

    if current is not None:
        rows.append(current)
    return rows


def write_csv(rows, output):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fields = [
        "target",
        "start_id",
        "sim_thr",
        "seed",
        "generated",
        "planned",
        "uniqueness_actual",
        "unique_over_planned",
        "success",
        "top_ds",
        "top_mol",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    if not rows:
        print("No lead-optimization task sections found.")
        return

    total = len(rows)
    success = sum(1 for r in rows if r["success"])
    print(f"Tasks parsed: {total}")
    print(f"Success rate: {100.0 * success / total:.2f}% ({success}/{total})")

    by_thr = defaultdict(lambda: [0, 0])
    by_target = defaultdict(lambda: [0, 0])
    for row in rows:
        by_thr[row["sim_thr"]][1] += 1
        by_thr[row["sim_thr"]][0] += int(row["success"])
        by_target[row["target"]][1] += 1
        by_target[row["target"]][0] += int(row["success"])

    print("\nSuccess by threshold:")
    for thr in sorted(by_thr):
        s, n = by_thr[thr]
        print(f"  {thr}: {100.0 * s / n:.2f}% ({s}/{n})")

    print("\nSuccess by target:")
    for target in sorted(by_target):
        s, n = by_target[target]
        print(f"  {target}: {100.0 * s / n:.2f}% ({s}/{n})")

    print("\nSuccessful top molecules:")
    for row in rows:
        if row["success"]:
            print(
                f"  {row['target']} id{row['start_id']} thr{row['sim_thr']}: "
                f"Top DS={row['top_ds']} | {row['top_mol']}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/csdnet_lead_all_0_live.log")
    parser.add_argument(
        "--output",
        default="CSDNet/exp/lead/results/lead_log_summary.csv",
    )
    args = parser.parse_args()

    rows = parse_log(args.log)
    write_csv(rows, args.output)
    print_summary(rows)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
