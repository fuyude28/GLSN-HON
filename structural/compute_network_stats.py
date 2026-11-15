# -*- coding: utf-8 -*-
# 一阶/高阶 GLSN 基础拓扑统计 (论文表格用)
# APL 与直径: 最大弱连通分量 (LWCC) 上按无向距离精确计算 (igraph C 实现)
# 聚集系数: 无向化后的平均局部聚集系数 (average local clustering coefficient)
import csv
from pathlib import Path
import igraph as ig

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data'

def load_igraph(path):
    E = []
    with open(path, encoding='utf-8') as f:
        r = csv.reader(f); next(r)
        for row in r:
            if len(row) >= 2:
                E.append((row[0], row[1]))
    nodes = sorted({x for e in E for x in e})
    idx = {v: i for i, v in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(idx[a], idx[b]) for a, b in E],
                 directed=True)
    return g

def stats(g):
    n, m = g.vcount(), g.ecount()
    out = {}
    out['N_nodes'] = n
    out['M_edges'] = m
    out['avg_total_degree'] = 2.0 * m / n
    out['density_directed'] = m / (n * (n - 1))
    gu = g.as_undirected(combine_edges=None)
    out['avg_clustering_coef'] = gu.transitivity_avglocal_undirected()
    strong_comps = g.connected_components(mode='strong')
    out['GSCC_size'] = max(map(len, strong_comps))
    out['n_strong_components'] = len(strong_comps)
    # 最大弱连通分量
    comps = g.connected_components(mode='weak')
    lwcc = comps.giant()
    lu = lwcc.as_undirected(combine_edges=None)
    out['avg_path_length'] = lu.average_path_length(directed=False)
    out['diameter'] = lu.diameter(directed=False)
    return out

rows = []
for name, path in [('First-order GLSN', DATA / 'network-fo.csv'),
                   ('Higher-order GLSN', DATA / 'network-ho.csv')]:
    print('processing', name, '...')
    out = stats(load_igraph(path))
    out['network'] = name
    rows.append(out)
    print({k: (round(v, 6) if isinstance(v, float) else v)
           for k, v in out.items()})

keys = ['network', 'N_nodes', 'M_edges', 'avg_total_degree',
        'density_directed', 'avg_clustering_coef',
        'avg_path_length', 'diameter', 'GSCC_size', 'n_strong_components']
with open(HERE / 'network_stats.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(keys)
    for r in rows:
        w.writerow([r[k] if not isinstance(r[k], float) else round(r[k], 6)
                    for k in keys])
print('saved', HERE / 'network_stats.csv')
