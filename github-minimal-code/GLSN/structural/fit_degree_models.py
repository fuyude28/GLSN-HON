# -*- coding: utf-8 -*-
"""Fit discrete degree tails on a shared support for each network/degree pair.

The cutoff is chosen by the discrete power-law KS criterion with at least
MIN_TAIL observations. Model comparison and goodness-of-fit are conditional on
that cutoff; the latter refits the selected model in each simulated tail.
"""

import csv
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, zeta

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data'
MIN_TAIL = 100
BOOTSTRAPS = 200
SEED = 42
MODELS = ('power_law', 'truncated_power_law', 'exponential')


def load_graph(path):
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        return nx.DiGraph((row[0], row[1]) for row in reader if len(row) >= 2)


def degree_values(graph, kind):
    values = (graph.degree() if kind == 'total' else
              graph.in_degree() if kind == 'in' else graph.out_degree())
    return np.fromiter((degree for _, degree in values), dtype=int)


def fit_power_law(x, xmin):
    n = len(x)
    sum_log = np.log(x).sum()

    def nll(alpha):
        return alpha * sum_log + n * np.log(zeta(alpha, xmin))

    result = minimize_scalar(nll, bounds=(1.000001, 10), method='bounded',
                             options={'xatol': 1e-9})
    return {'alpha': float(result.x), 'log_likelihood': -float(result.fun)}


def tpl_log_weights(xmin, alpha, rate, observed_max):
    # exp(-36) makes the omitted tail negligible at the fitted rate.
    upper = max(observed_max + 1, xmin + int(np.ceil(36 / rate)) + 1)
    k = np.arange(xmin, upper, dtype=float)
    return k, -alpha * np.log(k) - rate * k


def fit_truncated_power_law(x, xmin):
    n = len(x)
    sum_log = np.log(x).sum()
    sum_x = x.sum()
    observed_max = int(x.max())
    pl_alpha = fit_power_law(x, xmin)['alpha']

    def objective(theta):
        alpha, log_rate = theta
        rate = np.exp(log_rate)
        k, log_w = tpl_log_weights(xmin, alpha, rate, observed_max)
        log_z = logsumexp(log_w)
        p = np.exp(log_w - log_z)
        nll = alpha * sum_log + rate * sum_x + n * log_z
        gradient = (sum_log - n * np.dot(p, np.log(k)),
                    rate * (sum_x - n * np.dot(p, k)))
        return nll, np.asarray(gradient)

    starts = [(min(pl_alpha, 8), np.log(1 / max(x.mean(), 2))),
              (0.5, np.log(1 / max(x.mean() - xmin, 2)))]
    bounds = [(0, 10), (np.log(1e-4), np.log(5))]
    results = [minimize(objective, start, jac=True, bounds=bounds,
                        method='L-BFGS-B', options={'maxiter': 250, 'ftol': 1e-12})
               for start in starts]
    result = min(results, key=lambda item: item.fun)
    if not result.success and np.linalg.norm(result.jac, ord=np.inf) > 1e-3:
        raise RuntimeError(f'TPL optimization failed at xmin={xmin}: {result.message}')
    alpha, log_rate = result.x
    return {'alpha': float(alpha), 'rate': float(np.exp(log_rate)),
            'log_likelihood': -float(result.fun)}


def fit_exponential(x, xmin):
    mean_offset = float(x.mean() - xmin)
    rate = np.log1p(1 / mean_offset)
    log_likelihood = (len(x) * np.log(-np.expm1(-rate))
                      - rate * np.sum(x - xmin))
    return {'rate': rate, 'log_likelihood': float(log_likelihood)}


def fit_model(model, x, xmin):
    if model == 'power_law':
        return fit_power_law(x, xmin)
    if model == 'truncated_power_law':
        return fit_truncated_power_law(x, xmin)
    return fit_exponential(x, xmin)


def model_cdf(model, params, xmin, values):
    values = np.asarray(values, dtype=int)
    if model == 'power_law':
        return 1 - zeta(params['alpha'], values + 1) / zeta(params['alpha'], xmin)
    if model == 'exponential':
        return -np.expm1(-params['rate'] * (values - xmin + 1))
    k, log_w = tpl_log_weights(xmin, params['alpha'], params['rate'],
                               int(values.max()))
    cdf = np.cumsum(np.exp(log_w - logsumexp(log_w)))
    return cdf[values - xmin]


def model_ccdf(model, params, xmin, values):
    values = np.asarray(values, dtype=int)
    if model == 'power_law':
        return zeta(params['alpha'], values) / zeta(params['alpha'], xmin)
    if model == 'exponential':
        return np.exp(-params['rate'] * (values - xmin))
    k, log_w = tpl_log_weights(xmin, params['alpha'], params['rate'],
                               int(values.max()))
    weights = np.exp(log_w - logsumexp(log_w))
    survival = np.cumsum(weights[::-1])[::-1]
    return survival[values - xmin]


def ks_distance(x, model, params, xmin):
    ordered = np.sort(x)
    values = np.unique(ordered)
    # The supremum can fall in a gap immediately before the next observed k.
    points = np.union1d(values, values[values > xmin] - 1)
    empirical = np.searchsorted(ordered, points, side='right') / len(ordered)
    theoretical = model_cdf(model, params, xmin, points)
    return float(np.max(np.abs(empirical - theoretical)))


def choose_xmin(x):
    ordered = np.sort(x)
    candidates = np.unique(ordered)
    best = None
    for xmin in candidates:
        tail = ordered[ordered >= xmin]
        if len(tail) < MIN_TAIL:
            break
        fit = fit_power_law(tail, int(xmin))
        ks = ks_distance(tail, 'power_law', fit, int(xmin))
        if best is None or ks < best[0]:
            best = (ks, int(xmin), len(tail))
    return best[1]


def sample_tail(model, params, xmin, n, rng):
    if model == 'power_law':
        alpha = params['alpha']
        acceptance = zeta(alpha, xmin) / zeta(alpha, 1)
        samples = []
        remaining = n
        while remaining:
            batch = min(2_000_000, max(1000, int(1.2 * remaining / acceptance)))
            kept = rng.zipf(alpha, size=batch)
            kept = kept[kept >= xmin][:remaining]
            samples.append(kept)
            remaining -= len(kept)
        return np.concatenate(samples)
    if model == 'exponential':
        return xmin - 1 + rng.geometric(-np.expm1(-params['rate']), size=n)
    k, log_w = tpl_log_weights(xmin, params['alpha'], params['rate'], xmin)
    cdf = np.cumsum(np.exp(log_w - logsumexp(log_w)))
    cdf[-1] = 1.0
    return xmin + np.searchsorted(cdf, rng.random(n))


def fixed_xmin_gof(model, params, x, xmin, rng):
    observed = ks_distance(x, model, params, xmin)
    exceed = 0
    for _ in range(BOOTSTRAPS):
        simulated = sample_tail(model, params, xmin, len(x), rng)
        refit = fit_model(model, simulated, xmin)
        simulated_ks = ks_distance(simulated, model, refit, xmin)
        exceed += simulated_ks >= observed
    return (exceed + 1) / (BOOTSTRAPS + 1)


def analyze(network, kind, values, rng):
    positive = values[values > 0]
    xmin = choose_xmin(positive)
    tail = positive[positive >= xmin]
    rows = []
    for model in MODELS:
        fitted = fit_model(model, tail, xmin)
        n_params = 2 if model == 'truncated_power_law' else 1
        rows.append({
            'network': network, 'degree': kind, 'n_nodes': len(values),
            'n_zero': int(np.sum(values == 0)), 'xmin': xmin,
            'n_tail': len(tail), 'model': model,
            'alpha': fitted.get('alpha', ''), 'rate': fitted.get('rate', ''),
            'kc': 1 / fitted['rate'] if model == 'truncated_power_law' else '',
            'log_likelihood': fitted['log_likelihood'],
            'AIC': 2 * n_params - 2 * fitted['log_likelihood'],
            'KS': ks_distance(tail, model, fitted, xmin),
        })
    winner = min(rows, key=lambda item: item['AIC'])
    gof = fixed_xmin_gof(winner['model'], winner, tail, xmin, rng)
    for row in rows:
        row['delta_AIC'] = row['AIC'] - winner['AIC']
        row['best_AIC'] = row is winner
        row['gof_p_fixed_xmin'] = gof if row is winner else ''
        row['gof_reps'] = BOOTSTRAPS if row is winner else ''
    print(f"{network:3} {kind:5} xmin={xmin:3} n={len(tail):5} "
          f"best={winner['model']:20} KS={winner['KS']:.4f} "
          f"p_fixed_xmin={gof:.3f}", flush=True)
    return rows


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for network, filename in [('FO', 'network-fo.csv'), ('HON', 'network-ho.csv')]:
        graph = load_graph(DATA / filename)
        for kind in ('total', 'in', 'out'):
            rows.extend(analyze(network, kind, degree_values(graph, kind), rng))
    fields = ['network', 'degree', 'n_nodes', 'n_zero', 'xmin', 'n_tail',
              'model', 'alpha', 'rate', 'kc', 'log_likelihood', 'AIC', 'delta_AIC',
              'KS', 'best_AIC', 'gof_p_fixed_xmin', 'gof_reps']
    with open(HERE / 'degree_model_fits.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('saved', HERE / 'degree_model_fits.csv')


if __name__ == '__main__':
    main()
