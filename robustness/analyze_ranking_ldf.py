"""Compare Section 4.1 full-AUC ranks with fixed-attacker link-addition ranks."""

from __future__ import annotations

import json
import statistics
from ranking_ldf_experiment import CRITICAL, HERE, RANKING_SOURCES, read_csv, write_csv


def ranks(values: dict[str, float], reverse: bool = False) -> dict[str, float]:
    ordered = sorted(values.values(), reverse=reverse)
    return {method: statistics.mean(i + 1 for i, v in enumerate(ordered) if v == value)
            for method, value in values.items()}


def main() -> None:
    results = HERE / "results/ranking_ldf"
    full_auc = {}
    for method, (path, selector, _) in RANKING_SOURCES.items():
        if method == "DND-DRL":
            details = json.loads((CRITICAL / "DND-DRL/results/formal_real_summary.json").read_text(encoding="utf-8"))
            full_auc[method] = details["auc"]
        else:
            rows = [row for row in read_csv(path)
                    if selector is None or row["method"] == selector]
            full_auc[method] = float(rows[-1]["cumulative_auc"])
    dismantling_rank = ranks(full_auc)
    summary = read_csv(results / "summary.csv")
    comparisons, correlations = [], []
    for budget in sorted({int(row["budget"]) for row in summary}):
        gains = {row["method"]: float(row["mean_delta_r"]) for row in summary
                 if int(row["budget"]) == budget and row["method"] in full_auc}
        addition_rank = ranks(gains, reverse=True)
        for method in full_auc:
            comparisons.append(dict(method=method, budget=budget,
                                    dismantling_full_auc=full_auc[method],
                                    dismantling_rank=dismantling_rank[method],
                                    enhancement_delta_r=gains[method],
                                    enhancement_rank=addition_rank[method],
                                    rank_change=dismantling_rank[method] - addition_rank[method]))
        correlations.append(dict(budget=budget, methods=len(gains),
                                 spearman_rho=statistics.correlation(
                                     list(dismantling_rank.values()),
                                     [addition_rank[method] for method in dismantling_rank]),
                                 top_enhancement_method=max(gains, key=gains.get),
                                 dnd_drl_enhancement_rank=addition_rank["DND-DRL"]))
    write_csv(results / "ranking_comparison.csv", comparisons, list(comparisons[0]))
    write_csv(results / "ranking_correlation.csv", correlations, list(correlations[0]))
    final_budget = max(int(row["budget"]) for row in summary)
    final = sorted((row for row in comparisons if row["budget"] == final_budget),
                   key=lambda row: row["enhancement_rank"])
    random_row = next(row for row in summary
                      if row["method"] == "RandomNode" and int(row["budget"]) == final_budget)
    text = ["# Critical-node rankings and CB-LDF enhancement", "",
            "All ten methods use their saved Section 4.1 removal orders with the same current",
            "history-consistent bypass candidates, one-edge-per-state scan and CB-LDF choice.",
            "All methods receive identical budgets and the fixed original DND-DRL attacker.",
            "", f"## Results at B = {final_budget}", "",
            "| Ranking method | Full dismantling AUC | Dismantling rank | Enhancement delta R | Enhancement rank |",
            "|---|---:|---:|---:|---:|"]
    for row in final:
        text.append(f"| {row['method']} | {row['dismantling_full_auc']:.6f} | "
                    f"{row['dismantling_rank']:g} | {row['enhancement_delta_r']:.2%} | "
                    f"{row['enhancement_rank']:g} |")
    text.extend(["",
                 f"RandomNode-LDF: {float(random_row['mean_delta_r']):.2%} mean improvement, "
                 f"{float(random_row['sd_delta_r']):.2%} sample SD across {random_row['trials']} permutations.",
                 "Ranks compare only the same ten saved identification methods; rank 1 is best.",
                 "Lower full dismantling AUC and higher enhancement delta R are better.",
                 "", "## Budget dependence", "",
                 "| Budget | Highest enhancement | DND-DRL rank | Spearman rank correlation |",
                 "|---|---|---:|---:|"])
    for row in correlations:
        text.append(f"| {row['budget']} | {row['top_enhancement_method']} | "
                    f"{row['dnd_drl_enhancement_rank']:g} | {row['spearman_rho']:.3f} |")
    text.extend(["", "## Limitations", "",
                 "The fixed DND-DRL attacker may favor bypasses around its own priority states.",
                 "These results are conditional on this network, attack schedule and CB-LDF rule.",
                 "The saved full dismantling AUC and the 16-round enhancement area have different",
                 "horizons and sampling grids; only their method rankings are compared.",
                 "The ten saved rankings are single deterministic sequences. RandomNode SD",
                 "describes permutation variability and is not a confidence interval for the",
                 "deterministic differences. Rank correlations are descriptive across ten methods.",
                 ""])
    (results / "analysis.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({"B400": final, "budget_correlations": correlations}, indent=2))


if __name__ == "__main__":
    main()
