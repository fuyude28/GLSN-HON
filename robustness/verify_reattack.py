"""Verify identical attack IDs, fixed-sequence GSCC curves and summaries."""

import json
import math
import statistics
from collections import defaultdict

from reattack_experiment import (
    OUT, GLSN, SEQUENCE, BUDGETS, RANDOM_SEEDS, read_csv, load_edge_sets,
    read_graph, trapezoid,
)
from robustness_experiment import DATA, evaluate


def close(a, b):
    assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12), (a,b)


def main():
    labels,base_edges=read_graph(DATA/"network-ho.csv")
    index={label:i for i,label in enumerate(labels)};n=len(labels)
    original=read_csv(SEQUENCE)
    original_nodes=[r["state_node"] for r in original]
    order=[index[label] for label in original_nodes[1:]]
    baseline_curve=[int(r["gscc"]) for r in original]
    edge_sets=load_edge_sets(labels,base_edges)
    trials=read_csv(OUT/"trials.csv")
    expected={("Original GLSN",0,0)}
    expected.update((method,b,seed) for b in BUDGETS
                    for method in ("DND-DRL-LDF","RandomNode-LDF")
                    for seed in (RANDOM_SEEDS if method=="RandomNode-LDF" else (0,)))
    assert len(trials)==len(expected)==23
    assert {(r["method"],int(r["budget"]),int(r["seed"])) for r in trials}==expected
    baseline=next(r for r in trials if r["method"]=="Original GLSN")
    sampled_q=list(range(0,2145,math.ceil(n*0.01)))
    checkpoints=sampled_q+[len(order)]
    active_sets=[]
    removed=set();previous=0
    for q in checkpoints:
        removed.update(order[previous:q]);previous=q
        active_sets.append([node for node in range(n) if node not in removed])
    sampled_groups=defaultdict(list)
    for r in read_csv(OUT/"sampled_curves.csv"):
        sampled_groups[r["method"],int(r["budget"]),int(r["seed"])].append(r)
    assert set(sampled_groups)==expected
    groups=defaultdict(list)
    checked=0
    for row in trials:
        method,budget,seed=row["method"],int(row["budget"]),int(row["seed"])
        key=method,budget,seed
        curve=read_csv(OUT/"runs"/f"{method}_B{budget}_seed{seed}.csv")
        assert [r["state_node"] for r in curve]==original_nodes,key
        assert [int(r["removed"]) for r in curve]==list(range(len(order)+1))
        gscc=[int(r["gscc"]) for r in curve]
        assert all(a>=b for a,b in zip(gscc,gscc[1:])),key
        assert all(a>=b for a,b in zip(gscc,baseline_curve)),key
        for r in curve:close(r["gscc_fraction"],int(r["gscc"])/n)
        edges=base_edges if budget==0 else base_edges+edge_sets[method,seed][:budget]
        assert len(edges)==len(base_edges)+budget==int(row["edges"])
        _,actual=evaluate(n,edges,checkpoints,active_sets)
        assert actual==[gscc[q] for q in checkpoints],key
        checked+=len(checkpoints)
        assert int(row["removals"])==len(order)
        assert int(row["gscc_at_sequence_end"])==gscc[-1]
        assert int(row["gscc_at_2144"])==gscc[2144]
        samples=sampled_groups[key]
        assert [int(r["removed"]) for r in samples]==sampled_q
        assert [int(r["gscc"]) for r in samples]==[gscc[q] for q in sampled_q]
        close(row["sequence_auc"],sum(gscc[1:])/(n*n))
        close(row["partial_auc"],trapezoid([gscc[q] for q in sampled_q],sampled_q,n))
        for metric in ("sequence","partial"):
            close(row[f"{metric}_delta_r"],float(row[f"{metric}_auc"])/float(baseline[f"{metric}_auc"])-1)
        groups[method,budget].append(row)
    summary=read_csv(OUT/"summary.csv")
    assert len(summary)==len(groups)==5
    for row in summary:
        group=groups[row["method"],int(row["budget"])]
        assert int(row["runs"])==len(group)
        for column in row:
            if column.startswith("mean_"):
                key=column[5:];values=[float(r[key]) for r in group]
                close(row[column],statistics.mean(values))
                if len(values)>1:close(row[f"sd_{key}"],statistics.stdev(values))
                else:assert row[f"sd_{key}"]==""
    previous=read_csv(OUT/"archive/adaptive_three_conditions/trials.csv")
    previous={(r["method"],r["budget"],r["seed"]):r for r in previous}
    for row in trials:
        close(row["partial_auc"],previous[row["method"],row["budget"],row["seed"]]["fixed_partial_auc"])
    result=dict(passed=True,runs=len(trials),identical_full_sequences=23,
                shared_removals=len(order),independent_gscc_checkpoints=checked,
                partial_auc_matches_previous_fixed_attack=True,summary_rows=len(summary))
    (OUT/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":
    main()
