# Higher-order network modeling, dismantling and robustness enhancement

Minimal research code for **a higher-order network framework for improving
structural robustness in a directed transport network**.

The workflow has three modules: first-order / higher-order network construction,
DND-DRL critical-state identification, and history-consistent bypass link addition.

## Contents

```text
RESS-minimal/
├── data/
│   ├── network-ho.csv
│   └── network-fo.csv
├── modeling/
│   ├── build_networks.py
│   └── pyhon/
│       ├── BuildRulesFastParameterFreeFreq.py
│       ├── BuildNetwork.py
│       └── LICENSE.txt
├── dismantling/
│   ├── train.py
│   ├── identify.py
│   ├── protect.py
│   └── dnd_drl/                 # Dir-GNN, DQN, replay, environment and graph code
├── enhancement/
│   ├── link_addition.py
│   └── port_benefits.py
├── requirements.txt
├── README.md
└── .gitignore
```

The release contains exactly two dataset files. Runtime checkpoints, rankings,
logs and numerical outputs are written under `outputs/`, which is ignored by Git.
Run the commands below from this directory. Default data and output paths are
resolved relative to the scripts, so the folder can be relocated.

## Environment

Use Python 3.12 and an isolated environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then install:

```bash
python -m pip install -r requirements.txt
```

Dependencies are NumPy, NetworkX, igraph and PyTorch. The pinned versions were
used for the verification below. Training selects CUDA automatically when
available; `--device cpu` explicitly selects CPU. For GPU training, use a PyTorch
build appropriate for the installed CUDA runtime. Full training and the
100-repeat enhancement experiment take substantially longer than the small
verification runs; runtime depends on the device.

## Data format

Both CSV files have a header and two columns, `FromNode` and `ToNode`.
Each row is one directed, unweighted edge; node labels must be read as strings.

| File | Nodes | Directed edges | Initial GSCC |
| --- | ---: | ---: | ---: |
| `network-fo.csv` | 2,758 | 30,855 | 2,683 |
| `network-ho.csv` | 13,326 | 27,214 | 9,461 |

- `network-fo.csv` represents the first-order physical-port network.
- `network-ho.csv` represents the variable-order state network.
- `p|` denotes a first-order state at port `p`.
- `p|q.r` denotes current port `p`, preceded by `q`, then `r` going backward in
  history; its chronological port sequence is `r -> q -> p`.
- Physical ports are identified by the substring before `|`.

The networks derive from the 2025 container-vessel port-call data described in
the manuscript. Port names and coordinates are not needed by these algorithms.

## 1. Modeling

The supplied network CSVs can be used directly. To construct networks from your
own port-call sequences, prepare a headered CSV whose first column is the vessel
ID and whose second column is a hyphen-separated sequence of port IDs:

```text
vessel_id,trajectory
vessel_1,10-20-30-10
vessel_2,40-20-50-40
```

```bash
python modeling/build_networks.py --input /path/to/trajectories.csv --max-order 10 --min-support 10
```

Consecutive repeated ports are collapsed, matching the original construction
script. First-order edges are the unique adjacent transitions. Higher-order
rules and rewiring use the two bundled pyHON modules. Outputs are
`outputs/modeling/network-fo.csv` and `outputs/modeling/network-ho.csv`.
The two illustrative rows above explain the format; a meaningful higher-order
fit with minimum support 10 requires sufficient repeated observations.

## 2. Dismantling / critical-state identification

Train on synthetic directed Barabasi-Albert graphs:

```bash
python dismantling/train.py
```

Defaults follow the manuscript's setup: 30–100 nodes, two BA attachments,
reciprocity 0.1, hub-source bias 0.6, hidden dimension 64, three message-passing
layers, Adam learning rate 0.0001, batch size 64, five-step returns, discount
factor 1, replay capacity 500,000 and seed 42. After 50 warm-up episodes,
epsilon decreases from 1 to 0.05 over 5,000 episodes. The target network updates
every 1,000 optimization steps.

Validation uses 48 graphs with seed 30,000 every 1,000 episodes; early stopping
uses patience 5, minimum improvement 0.00015 and a minimum of 8,000 episodes.
The maximum is 30,000 episodes. An independent test set contains 100 graphs
with seed 40,000. The best validation checkpoint is saved to
`outputs/dismantling/best.pt`.

Apply that checkpoint to the higher-order network:

```bash
python dismantling/identify.py
```

This writes `outputs/dismantling/removal_sequence.csv` and `summary.json`.
The ranking contains states actually removed before all nontrivial SCCs
disappear. Each inference round preselects 1% of the remaining states, rounded
up and capped by the candidate set; candidates belong to the current largest
SCCs. GSCC size is measured after each removal. Dismantling AUC is
`sum(GSCC after each removal) / initial_N**2`.

Optional protection experiment using the generated ranking:

```bash
python dismantling/protect.py
```

Protection strategies are DND-DRL, random protection and no protection.
Targeted attacks share the same DND-DRL order; random attacks share the same
seed within each repeat. Defaults are 16 rounds, protecting and attacking
1% of the initial state count per round, with five random repeats. Outputs go
to `outputs/dismantling/protection/`.

## 3. Robustness enhancement

Run the global link-addition comparison:

```bash
python enhancement/link_addition.py
```

Strategies are CB-LDF, CB-LP, CB-HDF, CB-HP, Bypass Random and Global Random.
Budgets are 50, 100, ..., 400 links. Deterministic strategies run once; each
stochastic strategy uses 100 seeds starting at 42. The five bypass strategies
use the longest existing history-suffix match and add at most one link per
ranked critical state. Global Random samples missing directed pairs.
The attack horizon is 16 fixed batches of `ceil(0.01 * initial_N)` removals.
Outputs go to `outputs/enhancement/global/`.

Run independent physical-port interventions:

```bash
python enhancement/port_benefits.py
```

CB-LDF uses budgets of 1, 3 and 5 links per port. Bypass Random uses 20 repeats
at the main budget of 3. Both global and external targeted attacks are evaluated;
external attacks skip all states of the focal port. Eligibility and first-order
degrees are calculated directly from the two network CSVs. Outputs go to
`outputs/enhancement/ports/`. Use `--ports 1745,1387` to select port IDs,
subject to the same eligibility rules.

All six entry points expose their arguments through `--help`.

## Verification

Verified on 2026-11-04 with Python 3.12.7 and the pinned dependencies, using a
temporary copy outside the original project and a different working directory:

- Constructed FO and HO networks from synthetic trajectories.
- Completed four training episodes, including four optimizer updates,
  validation, checkpoint selection, checkpoint reload and held-out evaluation.
- Applied the fresh checkpoint to the supplied HO network for 300 removals.
- Ran all six global link-addition strategies at budgets 1 and 3, with two
  stochastic repeats and one attack round; checked history consistency.
- Ran both port-level attack scenarios for port 1813, including CB-LDF and
  Bypass Random; direct FO degrees matched the original saved degree table.
- Ran targeted and random attacks with all three protection strategies.
- Confirmed both packaged CSVs are byte-for-byte copies of the source networks.

Verification outputs and synthetic inputs were confined to the temporary copy.

## Implementation provenance

DND-DRL is retained from the upstream implementation. Training no longer loads
a diagnostic checkpoint or comparison scripts.
Inference is relocated from `evaluate_dnd.py`. Link addition and port benefits
retain the corresponding upstream experiment logic. Port eligibility is
computed inline using the upstream `preflight.py` rule. Protection retains the
upstream experiment dynamics for DND-DRL and the two controls.
Changes concern entry points, paths and removal of dependencies on saved results.

The two pyHON files are by Jian Xu and retain their upstream MIT license in
`modeling/pyhon/LICENSE.txt`.

## Limitations and implementation conventions

- Raw vessel trajectories are not distributed. The edge lists support
  dismantling and enhancement but cannot recover the frequencies needed to
  repeat construction and predictive comparisons. Modeling requires a
  separately supplied trajectory file.
- This core-method release excludes comparison implementations such as FINDER,
  DND, TAD and CoreHD, pretrained weights, saved experiment outputs and plotting
  scripts. Train a checkpoint before identification. Full training and all
  manuscript experiments have not been rerun for packaging.
- Original pyHON tests KL divergence from the extended distribution to the
  retained distribution, with threshold `new_order / log2(1 + support)`.
  Minimum support is applied to individual source-successor counts before
  normalization. These conventions differ from the KL direction, denominator
  and sequence-support wording in manuscript Eq. (3) / Section 2.2.
  The existing implementation is preserved.
- Enhancement uses trapezoidal partial areas at fixed attack-batch endpoints,
  including the intact network, as in the original experiment scripts.
  This differs from the per-removal discrete sum in manuscript Eq. (14).
  Dismantling uses the per-removal discrete AUC stated above.
- A newly trained ranking must supply enough states for the requested horizon,
  including after focal-port states or protected states are excluded.
  Scripts report insufficient sequences; reduce `--attack-rounds` or
  protection `--rounds` when conducting a shorter run.
- A release license for project-authored code and network data has not been
  specified. The MIT notice applies to the vendored pyHON files.
