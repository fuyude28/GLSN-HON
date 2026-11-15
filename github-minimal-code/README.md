# GLSN Minimal Reproducible Code

This folder contains the analysis and algorithm source code with the three supplied input datasets. Generated figures, experiment outputs, caches, checkpoints, and intermediate parameter/result tables are not included.

## Data

- `data/network-fo.csv`: first-order network edge list.
- `data/network-ho.csv`: higher-order network edge list.
- Physical-port system ID/name mapping CSV in `data/`.

## Environment

Python 3.10 or newer is recommended. Install the dependencies with:

```sh
python -m pip install -r requirements.txt
```

Training and evaluation commands are run from the repository root. For example:

```sh
python GLSN/critical/run_baselines.py --method dynamic_total_degree
python GLSN/critical/run_structural.py --method corehd
python GLSN/structural/compute_network_stats.py
python -m pytest GLSN/critical/tests
```

DND-DRL and FINDER evaluation require model checkpoints. Train a checkpoint first with the corresponding `train_dnd.py` or `train_nd.py`; checkpoints and run outputs are created locally and are not part of this package. The port-level result analyses from the full working directory also consume previously generated removal rankings and result tables, so they are not included in this minimal package.
