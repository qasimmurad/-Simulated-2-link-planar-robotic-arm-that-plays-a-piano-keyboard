# Simulated 2-Link Planar Robotic Arm That Plays a Piano Keyboard

This project simulates a 2-link planar robotic manipulator that plays melodies on a scaled piano keyboard. It sits at the intersection of **robot kinematics**, **motion planning**, and **symbolic music reasoning**, combining geometric IK/FK with graph search to find efficient joint-space trajectories across piano keys.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Setup and Installation](#3-setup-and-installation)
4. [Source Code Walkthrough](#4-source-code-walkthrough)
   - [4.1 Keyboard Layout (`data/keyboard_layout.py`)](#41-keyboard-layout)
   - [4.2 Kinematics (`src/robotics/kinematics.py`)](#42-kinematics)
   - [4.3 Workspace Analysis (`src/robotics/workspace.py`)](#43-workspace-analysis)
   - [4.4 Music Resolver (`src/music/resolver.py`)](#44-music-resolver)
   - [4.5 Prolog Knowledge Base (`src/music/kb.pl`)](#45-prolog-knowledge-base)
   - [4.6 Planning Heuristics (`src/planning/heuristics.py`)](#46-planning-heuristics)
   - [4.7 Motion Planning (`src/planning/search.py`)](#47-motion-planning)
   - [4.8 Visualisation (`src/viz/`)](#48-visualisation)
   - [4.9 Main Entry Point (`main.py`)](#49-main-entry-point)
5. [Running the Simulation](#5-running-the-simulation)
6. [Running the Tests](#6-running-the-tests)
7. [Experiments](#7-experiments)
8. [Design Decisions and Trade-offs](#8-design-decisions-and-trade-offs)

---

## 1. Project Overview

The simulated arm has two rigid links (L1 = 0.20 m, L2 = 0.15 m) with a fixed base centred above a single-octave keyboard (C4–C5). Given a melody as a sequence of note names, the system:

1. Resolves each note name to a physical (x, y) key position.
2. Runs inverse kinematics (IK) to find the joint angles (θ₁, θ₂) that place the arm tip on that key.
3. Uses a motion planner (greedy or A\*) to sequence the joint configurations so that total joint travel is minimised.
4. Animates the arm moving between keys.

The coordinate system has the arm base at the origin, with the keyboard lying along the positive-y direction at y = 0.30 m.

```
        BASE (0.0805, 0)
             |
    link 1   |  θ₁
       ______|
      |
      |_____ link 2  θ₂ (relative to link 1)
             |
             tip → strikes key at (x, 0.30)
```

---

## 2. Repository Structure

```
.
├── data/
│   └── keyboard_layout.py      # Physical key positions and arm parameters
├── src/
│   ├── music/
│   │   ├── kb.pl               # Prolog knowledge base (notes, intervals, melodies)
│   │   └── resolver.py         # Maps note names → (x, y) positions
│   ├── planning/
│   │   ├── heuristics.py       # Joint-travel cost and A* heuristic
│   │   └── search.py           # Greedy and A* motion planners
│   ├── robotics/
│   │   ├── kinematics.py       # Forward and inverse kinematics
│   │   └── workspace.py        # Reachability queries
│   └── viz/
│       ├── animate.py          # Animated arm simulation
│       └── plots.py            # Static workspace and trajectory plots
├── tests/
│   ├── test_kinematics.py      # FK/IK unit tests
│   └── test_search.py          # Planning and heuristics unit tests
├── experiments/
│   ├── run_search_comparison.py  # Greedy vs A* cost comparison
│   └── run_tempo_analysis.py     # Max playable BPM analysis
├── main.py                     # CLI entry point
├── requirements.txt
└── pytest.ini
```

---

## 3. Setup and Installation

**Python 3.9 or higher is required.**

```bash
# Clone the repository
git clone https://github.com/qasimmurad/-Simulated-2-link-planar-robotic-arm-that-plays-a-piano-keyboard.git
cd -Simulated-2-link-planar-robotic-arm-that-plays-a-piano-keyboard

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` installs: `numpy`, `matplotlib`, `pyswip` (Python–Prolog bridge), `pytest`.

> **Note on `pyswip`:** The Prolog knowledge base (`kb.pl`) is standalone and valid SWI-Prolog. The `pyswip` package is listed as a dependency for future Python↔Prolog integration; it is not required to run the simulation or tests.

---

## 4. Source Code Walkthrough

The recommended reading order follows the data flow: world model → kinematics → music → planning → visualisation.

---

### 4.1 Keyboard Layout

**File:** `data/keyboard_layout.py`

This is the world model. It defines the physical geometry of the keyboard and the arm parameters used everywhere else in the codebase.

```
WHITE_KEY_SPACING = 0.023 m   (standard piano key width)
WHITE_KEY_Y       = 0.30 m    (distance of keys from arm base along y-axis)

L1 = 0.20 m   (shoulder → elbow link length)
L2 = 0.15 m   (elbow → tip link length)
BASE = (3.5 × WHITE_KEY_SPACING, 0.0)   (arm base, centred above the keyboard)
```

Eight white keys (C4–C5) are spaced 23 mm apart along the x-axis. Five black keys sit 12 mm closer to the arm (smaller y). All positions are stored in `WHITE_KEYS`, `BLACK_KEYS`, and `ALL_KEYS` dictionaries keyed by note name (e.g. `"C4"`, `"F#4"`).

**Key design choice:** the arm base is placed at x = 3.5 × 23 mm = 80.5 mm, centred over the octave. This maximises symmetric reachability — the workspace annulus (r_min ≤ r ≤ r_max, where r_min = |L1−L2| = 0.05 m and r_max = L1+L2 = 0.35 m) covers all 13 keys.

---

### 4.2 Kinematics

**File:** `src/robotics/kinematics.py`

The mathematical core of the project. Contains four functions.

#### `forward_kinematics(theta1, theta2, base, l1, l2)`

Computes the (x, y) positions of the elbow and tip given joint angles:

```
elbow = BASE + L1 · [cos θ₁,  sin θ₁]
tip   = elbow + L2 · [cos(θ₁+θ₂),  sin(θ₁+θ₂)]
```

Returns a triple `(base, elbow, tip)` — all three points needed to draw the arm.

#### `inverse_kinematics(target, base, l1, l2, elbow_up)`

Given a target (x, y), finds joint angles using the **geometric (cosine-rule) method**:

1. Translate to arm-local coordinates: `dx = px − bx`, `dy = py − by`, `r = √(dx²+dy²)`
2. Check reachability: `|L1−L2| ≤ r ≤ L1+L2` (triangle inequality)
3. Solve θ₂ via the cosine rule: `cos θ₂ = (r²−L1²−L2²) / (2·L1·L2)`
4. Solve θ₁: `α = atan2(dy, dx)`, `β = arccos((r²+L1²−L2²) / (2·r·L1))`
   - Elbow-up:   `θ₁ = α − β`
   - Elbow-down: `θ₁ = α + β`

Returns `(theta1, theta2)` or `None` if the target is unreachable. `np.clip` guards against floating-point values slightly outside [−1, 1] before calling `arccos`.

#### `is_reachable(target, ...)`

Convenience wrapper around the triangle inequality check. Used by the workspace module and tests.

#### `choose_solution(target, current_theta1, ...)`

Resolves the IK ambiguity (elbow-up vs. elbow-down) by picking whichever solution requires **less shoulder-joint travel** from the current pose. This is the strategy used by the greedy planner.

---

### 4.3 Workspace Analysis

**File:** `src/robotics/workspace.py`

Thin utility layer on top of kinematics:

- `reachable_keys()` / `unreachable_keys()` — filter `ALL_KEYS` by reachability. All 13 keys in the current layout are reachable.
- `workspace_grid(resolution)` — samples the plane on an N×N grid and returns every point inside the reachable annulus. Used by visualisation to draw the workspace boundary.

---

### 4.4 Music Resolver

**File:** `src/music/resolver.py`

Bridges music notation and physical space. Defines three built-in melodies as Python lists of note name strings:

| Name | Notes |
|---|---|
| `twinkle` | Twinkle Twinkle Little Star (14 notes, C major) |
| `mary` | Mary Had a Little Lamb (13 notes) |
| `ode_to_joy` | Ode to Joy theme (15 notes) |

`resolve_note(note)` looks up a note name in `ALL_KEYS` with a case-insensitive fallback. `resolve_melody(notes)` maps a list of note names to `(note, position)` pairs, silently skipping any note not on the keyboard.

The output of `resolve_melody` is the direct input to the motion planners.

---

### 4.5 Prolog Knowledge Base

**File:** `src/music/kb.pl`

A SWI-Prolog knowledge base encoding music theory facts and rules:

- **Facts:** `midi/2` — maps each note name to its MIDI pitch number (e.g. `midi('C4', 60)`).
- **Rules:** `semitone_distance/3`, `step/2` (≤ 2 semitones), `leap/2` (> 2 semitones).
- **Scale:** `c_major/1` facts for all 8 diatonic notes; `in_scale/2` checks membership.
- **Melodies:** `melody/2` — the same three melodies as in `resolver.py`, queryable from Prolog.
- **`all_in_scale/2`** — recursively checks that every note in a list belongs to a scale.

This knowledge base supports symbolic reasoning about music (e.g. detecting leaps, verifying scale membership) independently of the Python simulation.

---

### 4.6 Planning Heuristics

**File:** `src/planning/heuristics.py`

Two functions used by the A\* planner:

#### `joint_travel_cost(t1a, t2a, t1b, t2b)`

The **L1 norm in joint space** — sum of absolute angle changes across both joints. This is the edge cost used throughout the planner. L1 is preferred over L2 because it penalises large individual-joint movements, which more closely reflects real actuation limits.

#### `joint_space_heuristic(current_angles, target_pos)`

The **admissible heuristic** for A\*. Calls `choose_solution` to find the cheapest IK solution to `target_pos` from the current configuration, then returns the joint travel to that solution. It is admissible because it is exactly the cost of the optimal single move to that target — it never overestimates.

---

### 4.7 Motion Planning

**File:** `src/planning/search.py`

The central algorithmic contribution. Treats melody playback as a **shortest-path problem in joint space**: given a fixed sequence of notes, find the assignment of IK solutions (elbow-up or elbow-down, for each note) that minimises total joint travel.

#### `greedy_plan(note_positions)`

At each step, calls `choose_solution` to pick the IK solution with least joint travel from the current configuration. O(n) in the number of notes. Fast, but myopic — a locally cheap move may force a costly one later.

#### `astar_plan(note_positions)`

A\* search over the IK solution choices. The state is `(step_index, theta1, theta2)`. At each step, both elbow-up and elbow-down IK solutions are expanded. The priority is `g + h`, where:

- `g` = total joint travel so far
- `h` = `joint_space_heuristic` to the *next* target (one-step lookahead)

A `visited` dict prunes states that have already been reached at lower cost. The result is the **globally optimal assignment** of IK solutions for the given melody, under the joint-travel cost metric.

In practice, for these short melodies over a single octave, greedy and A\* often produce identical plans — the keyboard is compact enough that local and global optima coincide. The A\* implementation is architecturally important: it demonstrates the correctness of the heuristic and provides a verified baseline.

#### `total_joint_travel(plan)`

Sums the `joint_travel_cost` between every consecutive pair of configurations. Used to compare plans and report results.

---

### 4.8 Visualisation

**File:** `src/viz/plots.py` — static plots  
**File:** `src/viz/animate.py` — animation

#### `plots.py`

- `plot_workspace(ax, show)` — draws the reachable annulus (filled), the arm base, and all 13 key positions labelled by note name. The outer circle has radius L1+L2 = 0.35 m; the inner circle (dead zone) has radius |L1−L2| = 0.05 m.
- `plot_arm(theta1, theta2, ax, ...)` — draws one arm configuration (three points connected by lines: base → elbow → tip).
- `plot_plan(plan, show)` — overlays the full plan on the workspace diagram, colouring each arm pose progressively from dark to light using the `viridis` colormap.

#### `animate.py`

- `_interpolate(t1a, t2a, t1b, t2b, steps)` — linearly interpolates between two joint configurations, producing smooth arm motion between keys.
- `_build_frames(plan, steps_per_move)` — flattens the plan into a list of `(theta1, theta2, note)` frames by interpolating between every consecutive pair.
- `animate_plan(plan, steps_per_move, interval, save_path)` — renders the animation with `matplotlib.animation.FuncAnimation`. The arm is drawn in red over a static background (workspace circle + key dots). The current note is displayed as a text overlay. If `save_path` is given, saves to `.gif` (Pillow) or `.mp4` (ffmpeg).

---

### 4.9 Main Entry Point

**File:** `main.py`

Command-line interface that wires all modules together:

```
python main.py [--melody {twinkle,mary,ode_to_joy}]
               [--planner {greedy,astar}]
               [--no-animate]
               [--workspace]
               [--save FILE.gif]
```

Execution flow:
1. Parse arguments
2. `resolve_melody(notes)` → list of `(note, pos)` pairs
3. `greedy_plan` or `astar_plan` → list of `(note, θ₁, θ₂)` configurations
4. Print the plan table and total joint travel
5. Optionally show `plot_workspace()`
6. `animate_plan(plan)` → interactive window or saved file

---

## 5. Running the Simulation

```bash
# Default: Twinkle Twinkle, A* planner, interactive animation
python main.py

# Play Mary Had a Little Lamb with the greedy planner
python main.py --melody mary --planner greedy

# Print the plan table without opening a window
python main.py --no-animate

# Show the workspace diagram first, then animate
python main.py --workspace

# Save animation as a GIF (requires Pillow)
python main.py --save twinkle.gif
```

Example output for `python main.py --no-animate`:

```
Melody : twinkle  (14 notes)
Planner: astar
Total joint travel: 1.1243 rad

Step  Note     theta1 (rad)   theta2 (rad)
------------------------------------------
0     C4             1.4236         0.9687
1     C4             1.4236         0.9687
2     G4             1.0730         1.0922
...
13    C4             1.4236         0.9687
```

---

## 6. Running the Tests

```bash
python -m pytest -v
```

Expected output:

```
tests/test_kinematics.py::test_fk_straight_out              PASSED
tests/test_kinematics.py::test_ik_then_fk_roundtrip         PASSED
tests/test_kinematics.py::test_unreachable_target           PASSED
tests/test_kinematics.py::test_is_reachable_all_white_keys  PASSED
tests/test_kinematics.py::test_elbow_up_and_down_both_valid PASSED
tests/test_search.py::test_greedy_plan_length               PASSED
tests/test_search.py::test_greedy_plan_returns_floats       PASSED
tests/test_search.py::test_astar_plan_length                PASSED
tests/test_search.py::test_astar_no_worse_than_greedy       PASSED
tests/test_search.py::test_joint_travel_zero_same_config    PASSED
tests/test_search.py::test_joint_travel_positive            PASSED
tests/test_search.py::test_greedy_raises_on_unreachable     PASSED
tests/test_search.py::test_astar_raises_on_unreachable      PASSED
tests/test_search.py::test_total_joint_travel_empty         PASSED
tests/test_search.py::test_total_joint_travel_single        PASSED

15 passed in 0.12s
```

**Kinematics tests** (`test_kinematics.py`) verify correctness at the mathematical level: FK with zero angles places the tip at BASE + L1 + L2 along the x-axis; the IK→FK roundtrip recovers the original target for every key; all white keys are reachable; both IK solutions exist for a mid-workspace target.

**Planning tests** (`test_search.py`) verify: plan length matches input; A\* cost is never worse than greedy; joint travel is zero for identical configs; `ValueError` is raised for unreachable notes.

---

## 7. Experiments

#### Search Comparison

```bash
python experiments/run_search_comparison.py
```

Prints a table comparing total joint travel for greedy vs. A\* across all three melodies. Demonstrates that A\* finds plans no worse than greedy and verifies the heuristic is admissible.

#### Tempo Analysis

```bash
python experiments/run_tempo_analysis.py
```

Estimates the maximum playable BPM for each melody, assuming a joint velocity limit of π rad/s. The bottleneck is the single largest joint-angle move in the plan: `max_BPM = 60 × π / max_move_cost`. Also identifies which note transition is the physical bottleneck.

---

## 8. Design Decisions and Trade-offs

| Decision | Rationale |
|---|---|
| Geometric IK (cosine rule) over numerical IK | Closed-form, exact, and fast for a 2-DOF planar arm. No convergence issues. |
| L1 joint-space cost over L2 | Penalises large single-joint swings; more representative of actuator limits than Euclidean joint-space distance. |
| A\* with one-step lookahead heuristic | Guarantees global optimality for this problem structure. The heuristic is admissible (never overestimates) because it computes the exact minimum cost to the very next target. |
| `choose_solution` minimises θ₁ deviation | The shoulder joint has a larger moment arm and typically dominates motion time; minimising shoulder travel is a practical proxy for speed. |
| Fixed melody sequences (no MIDI input) | Keeps the scope focused on planning. The architecture (resolver → planner → animator) is designed to accept any `List[str]` of note names, so MIDI parsing could be added as a new resolver module without changing downstream code. |
| Prolog KB alongside Python | Separates symbolic music reasoning (scale membership, interval classification) from numeric simulation. The KB is queryable independently and extensible to chord voicing, harmonisation, or constraint checking. |
