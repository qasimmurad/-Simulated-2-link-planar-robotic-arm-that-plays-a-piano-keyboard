# 2-Link Planar Robotic Arm Piano Player

A simulation of a 2-link planar robotic arm playing melodies on a piano
keyboard. The arm uses symbolic music reasoning and search-based motion
planning to find physically efficient key sequences.

Built for CS152: Harnessing Artificial Intelligence Algorithms,
Minerva University, Spring 2026.

## Architecture

```
[Music Layer]          [Search Layer]           [Robotics Layer]
src/music/             src/planning/             src/robotics/
resolver.py    →       search.py        →        kinematics.py
kb.pl                  heuristics.py             workspace.py
                                                 singularity.py
↓                    ↓                          ↓
Note sequences     Joint-angle plans         FK / IK / Jacobian
                         ↓
                     src/viz/
                     animate.py
                     plots.py
```

Data flows left to right: note names → (x,y) key positions → IK
configurations → A* planned sequence → matplotlib animation.

## Setup

```bash
pip install -r requirements.txt
```

SWI-Prolog is optional (only needed to query kb.pl directly).

## Usage

```bash
# Animate the arm playing Twinkle Twinkle (default)
python main.py

# Choose a different melody
python main.py --melody mary
python main.py --melody ode_to_joy

# Choose planning algorithm
python main.py --planner greedy
python main.py --planner astar_wide   # recommended

# Print plan without animating
python main.py --no-animate

# Save animation to file
python main.py --save twinkle.gif

# Run search comparison experiment (produces results/search_comparison.csv)
python experiments/run_search_comparison.py

# Generate workspace and trajectory figures
python experiments/generate_figures.py

# Run all tests
python -m pytest tests/ -v
```

## Learning Outcome Coverage

**#search** — Three planning algorithms implemented and benchmarked:
- Greedy best-first: locally optimal at each step, ~18% suboptimal overall
- UCS (Dijkstra): optimal, used as correctness baseline
- A* with wide state space: optimal with 14.9% fewer node expansions
  than UCS on Twinkle, 7.8% on Mary, 7.2% on Ode to Joy

Two heuristics analysed: joint-space L1 distance (admissible, used in
production) and Euclidean end-effector distance (inadmissible on wide
state space — counterexample documented in tests and docs/admissibility_proofs.md).

**#robotics** — Forward and inverse kinematics derived from first
principles using the law of cosines. Both elbow-up and elbow-down
configurations handled with smart configuration selection. Workspace
reachability analysis (see results/figures/). Jacobian derived
analytically; singularity detection and Yoshikawa manipulability measure
implemented in src/robotics/singularity.py.

**#aicoding** — Modular architecture with four independent layers
(music, planning, robotics, visualisation). 67 tests across
3 test files, all passing. Reproducible experiments producing CSV
outputs and publication-quality figures.

## Results

| Melody | Greedy cost | A* cost | Saving | A* nodes | UCS nodes |
|---|---|---|---|---|---|
| Twinkle | 1.1243 | 0.9151 | 18.6% | 57 | 67 |
| Mary | 1.0312 | 0.9100 | 11.8% | 47 | 51 |
| Ode to Joy | 1.0801 | 0.9717 | 10.0% | 64 | 69 |

Cost unit: total joint-angle travel in radians (sum of |Δθ₁| + |Δθ₂|
across all note transitions).

## Project Structure

```
.
├── data/
│   └── keyboard_layout.py     # Key (x,y) positions, arm parameters
├── docs/
│   └── admissibility_proofs.md
├── experiments/
│   ├── generate_figures.py
│   └── run_search_comparison.py
├── results/
│   ├── figures/
│   │   ├── workspace_reachability.png
│   │   └── twinkle_trajectory.png
│   └── search_comparison.csv
├── src/
│   ├── music/
│   │   ├── kb.pl              # Prolog music knowledge base
│   │   └── resolver.py        # Note → position lookup + melodies
│   ├── planning/
│   │   ├── heuristics.py      # Admissible heuristics + cost functions
│   │   └── search.py          # Greedy, UCS, A* planners
│   ├── robotics/
│   │   ├── kinematics.py      # FK, IK, configuration selection
│   │   ├── singularity.py     # Jacobian, singularity, manipulability
│   │   └── workspace.py       # Reachability analysis
│   └── viz/
│       ├── animate.py         # matplotlib animation
│       └── plots.py           # Static workspace + trajectory plots
├── tests/
│   ├── test_kinematics.py
│   ├── test_search.py
│   └── test_singularity.py
├── main.py
├── pytest.ini
└── requirements.txt
```
