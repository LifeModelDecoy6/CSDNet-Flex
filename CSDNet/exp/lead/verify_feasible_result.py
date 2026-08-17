import argparse
import csv
import math
from pathlib import Path


def verify_result(path, sim_threshold, min_calls, oracle_budget):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty Lead result: {path}")

    rows = []
    seen = set()
    with path.open(newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 5:
                raise RuntimeError(
                    f"Malformed row {line_number}: expected at least 5 columns"
                )
            smiles = row[0].strip()
            if not smiles:
                raise RuntimeError(f"Empty SMILES at row {line_number}")
            if smiles in seen:
                raise RuntimeError(f"Duplicate oracle molecule at row {line_number}")
            try:
                dock, qed, sa, similarity = map(float, row[1:5])
            except ValueError as exc:
                raise RuntimeError(f"Non-numeric metrics at row {line_number}") from exc
            if not all(math.isfinite(value) for value in (dock, qed, sa, similarity)):
                raise RuntimeError(f"Non-finite metrics at row {line_number}")
            if qed < 0.6 or sa < 6.0 / 9.0 or similarity < sim_threshold:
                raise RuntimeError(
                    f"Infeasible oracle row {line_number}: "
                    f"QED={qed:.6f} SA={sa:.6f} SIM={similarity:.6f}"
                )
            seen.add(smiles)
            rows.append(row)

    calls = len(rows)
    if calls < min_calls:
        raise RuntimeError(f"Too few docking calls: {calls}/{min_calls}")
    if calls > oracle_budget:
        raise RuntimeError(f"Oracle budget exceeded: {calls}/{oracle_budget}")
    return calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("--sim_threshold", type=float, required=True)
    parser.add_argument("--min_calls", type=int, default=500)
    parser.add_argument("--oracle_budget", type=int, default=1000)
    args = parser.parse_args()
    calls = verify_result(
        args.result,
        args.sim_threshold,
        args.min_calls,
        args.oracle_budget,
    )
    print(
        f"Verified feasible-only Lead result: calls={calls}, "
        f"threshold={args.sim_threshold}, path={args.result}"
    )


if __name__ == "__main__":
    main()
