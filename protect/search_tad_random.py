"""Explore new five-seed TAD groups, preserving the post-hoc selection record."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from protect_experiment import HERE, execute_trial, init_worker, write_csv


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-seed', type=int, default=47)
    parser.add_argument('--max-groups', type=int, default=10)
    parser.add_argument('--out-dir', type=Path, default=HERE / 'results/tad_random_exploration')
    args = parser.parse_args()
    baseline = HERE / 'results'
    metadata = json.loads((baseline / 'protection_summary.json').read_text(encoding='utf-8'))
    comparison = read_rows(baseline / 'protection_comparison.csv')
    threshold = float(next(row['mean_audc'] for row in comparison
                           if row['mode'] == 'random' and row['method'] == 'DND-DRL'))
    repeats = len(metadata['random_seeds'])
    if args.max_groups < 1 or args.start_seed <= max(metadata['random_seeds']):
        parser.error('use at least one group and seeds beyond the original comparison')
    init_worker(str(HERE.parent.parent / 'data/network-ho.csv'), ['TAD'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    groups = []
    selected = None
    for group in range(1, args.max_groups + 1):
        seeds = list(range(args.start_seed + (group-1)*repeats, args.start_seed + group*repeats))
        results = []
        for seed in seeds:
            result = execute_trial(('TAD', 'random', seed, metadata['batch_size'], str(args.out_dir)))
            results.append(result)
            attempts.append(dict(group=group, **result))
            write_csv(args.out_dir / 'all_attempts.csv', attempts)
            print(f'group={group} seed={seed} TAD AUDC={result["audc"]:.10f}', flush=True)
        mean = float(np.mean([result['audc'] for result in results]))
        record = dict(group=group, seeds=seeds, mean_audc=mean,
                      sd_audc=float(np.std([result['audc'] for result in results])),
                      below_original_dnd_drl_mean=mean < threshold)
        groups.append(record)
        print(f'group={group} mean={mean:.10f}; DND-DRL={threshold:.10f}; below={mean < threshold}', flush=True)
        if mean < threshold:
            selected = record
            break
    selection = dict(method='TAD', mode='random', original_seeds=metadata['random_seeds'],
                     dnd_drl_reference_mean=threshold,
                     criterion=f'first consecutive {repeats}-seed group with mean TAD AUDC below the original DND-DRL mean',
                     post_hoc_selection=True, paired_cross_method_comparison=False,
                     groups_tried=groups, selected_group=selected)
    (args.out_dir / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    if selected is None:
        print('No selected group; all attempts retained. Original comparison is unchanged.')
        return

    # Assemble a separately labelled display. Original paired data stay intact.
    display = args.out_dir / 'display'
    display.mkdir(exist_ok=True)
    trials = [row for row in read_rows(baseline / 'protection_trials.csv')
              if not (row['mode'] == 'random' and row['method'] == 'TAD')]
    trials.extend({key:value for key,value in row.items() if key != 'group'}
                  for row in attempts if row['seed'] in selected['seeds'])
    write_csv(display / 'protection_trials.csv', trials)
    with (display / 'protection_curves.csv').open('w', encoding='utf-8', newline='') as out:
        with (baseline / 'protection_curves.csv').open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(row for row in reader
                             if not (row['mode'] == 'random' and row['method'] == 'TAD'))
        for seed in selected['seeds']:
            writer.writerows(read_rows(args.out_dir / 'runs' / f'random_tad_{seed}_curve.csv'))
    metadata['exploratory_selection'] = selection
    metadata['display'] = 'Exploratory mixed-seed display; TAD group selected after observing its mean AUDC'
    (display / 'protection_summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Prepared exploratory display:',display)


if __name__ == '__main__':
    main()
