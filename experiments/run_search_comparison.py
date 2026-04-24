"""
run_search_comparison.py

Full comparison of motion-planning strategies across all melodies.

Runs:
  - greedy_plan              (no search, no heuristic)
  - ucs_plan                 (Dijkstra, h=0 — establishes the verified optimum)
  - A* with joint-space heuristic  (_wide_heuristic, admissible)
  - A* with euclidean heuristic    (h_euclidean_endeffector, inadmissible for
                                    wide configs — see Task 4 findings)
  - A* with combined heuristic     (_h_combined_wide, inadmissible for same reason)

Outputs:
  stdout  — formatted table with Optimal? column (✓/✗ vs UCS cost)
  results/search_comparison.csv

Run from the project root:
    python experiments/run_search_comparison.py
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.music.resolver import resolve_melody, MELODIES
from src.planning.search import (
    greedy_plan,
    ucs_plan_instrumented,
    astar_plan_wide_instrumented,
    astar_plan_wide_instrumented_euclidean,
    astar_plan_wide_instrumented_combined,
    PlanResult,
    total_joint_travel,
)

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "search_comparison.csv")
CSV_FIELDS = ["melody", "notes", "algorithm", "heuristic", "admissible", "cost", "optimal", "nodes", "runtime_ms"]

ADMISSIBLE_TAG  = "yes"
INADMISSIBLE_TAG = "no (wide-config perturbed-tip counterexample)"


def _greedy_result(positions) -> PlanResult:
    t0 = time.perf_counter()
    plan = greedy_plan(positions)
    ms = (time.perf_counter() - t0) * 1_000.0
    return PlanResult(plan=plan, nodes_expanded=0, runtime_ms=ms)


def main():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # ------------------------------------------------------------------ #
    # Collect rows
    # ------------------------------------------------------------------ #
    rows = []

    for melody_name, notes in MELODIES.items():
        positions = resolve_melody(notes)
        n_notes   = len(positions)

        # UCS establishes the verified optimal cost for this melody.
        ucs = ucs_plan_instrumented(positions)
        optimum = total_joint_travel(ucs.plan)

        def is_optimal(cost: float) -> bool:
            return abs(cost - optimum) < 1e-6

        def tick(label: str, heuristic: str, admissible: str, result: PlanResult):
            cost = total_joint_travel(result.plan)
            rows.append({
                "melody":      melody_name,
                "notes":       n_notes,
                "algorithm":   label,
                "heuristic":   heuristic,
                "admissible":  admissible,
                "cost":        round(cost, 6),
                "optimal":     "yes" if is_optimal(cost) else "no",
                "nodes":       result.nodes_expanded,
                "runtime_ms":  round(result.runtime_ms, 3),
            })

        tick("greedy",  "none",      "n/a",              _greedy_result(positions))
        tick("ucs",     "none (h=0)", ADMISSIBLE_TAG,    ucs)
        tick("astar",   "joint-space (wide, 6 configs)", ADMISSIBLE_TAG,
             astar_plan_wide_instrumented(positions))
        tick("astar",   "euclidean end-effector", INADMISSIBLE_TAG,
             astar_plan_wide_instrumented_euclidean(positions))
        tick("astar",   "combined (max of above two)", INADMISSIBLE_TAG,
             astar_plan_wide_instrumented_combined(positions))

    # ------------------------------------------------------------------ #
    # Print formatted table
    # ------------------------------------------------------------------ #
    col_w = {
        "melody":    12,
        "algorithm": 7,
        "heuristic": 30,
        "cost":      8,
        "optimal":   8,
        "nodes":     7,
        "runtime_ms": 10,
    }

    header = (
        f"{'Melody':<{col_w['melody']}}  "
        f"{'Alg':<{col_w['algorithm']}}  "
        f"{'Heuristic':<{col_w['heuristic']}}  "
        f"{'Cost':>{col_w['cost']}}  "
        f"{'Optimal?':>{col_w['optimal']}}  "
        f"{'Nodes':>{col_w['nodes']}}  "
        f"{'ms':>{col_w['runtime_ms']}}"
    )
    separator = "-" * len(header)
    print(header)
    print(separator)

    prev_melody = None
    for r in rows:
        melody_col = r["melody"] if r["melody"] != prev_melody else ""
        prev_melody = r["melody"]
        optimal_sym = "✓" if r["optimal"] == "yes" else "✗"
        heuristic_display = r["heuristic"][:col_w["heuristic"]]
        nodes_display = str(r["nodes"]) if r["nodes"] > 0 else "—"
        print(
            f"{melody_col:<{col_w['melody']}}  "
            f"{r['algorithm']:<{col_w['algorithm']}}  "
            f"{heuristic_display:<{col_w['heuristic']}}  "
            f"{r['cost']:>{col_w['cost']}.4f}  "
            f"{optimal_sym:>{col_w['optimal']}}  "
            f"{nodes_display:>{col_w['nodes']}}  "
            f"{r['runtime_ms']:>{col_w['runtime_ms']}.2f}"
        )
        if melody_col == "" and r["algorithm"] == "astar" and "combined" in r["heuristic"]:
            print()

    print(separator)
    print()
    print("Notes:")
    print("  Optimal? = cost matches UCS-verified optimum (tolerance 1e-6)")
    print("  Euclidean and combined heuristics are inadmissible for the wide-search")
    print("  state space: perturbed configs (theta1 ± 0.1 rad) do not land exactly")
    print("  at the target position, so h_euclidean can overestimate (Task 4 finding).")
    print("  joint-space heuristic (_wide_heuristic) is admissible and always optimal.")

    # ------------------------------------------------------------------ #
    # Write CSV
    # ------------------------------------------------------------------ #
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written → {CSV_PATH}")


if __name__ == "__main__":
    main()
