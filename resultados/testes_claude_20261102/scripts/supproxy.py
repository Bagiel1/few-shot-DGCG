"""Diagnóstico: a homofilia das arestas ENTRE SUPORTES (rótulos conhecidos) prevê a das consultas?"""
import numpy as np, torch, json
import fs_lab as L
from proto_sgc.episodes import FeatureEpisodeSampler, split_classes
out = []
for name in L.NAMES:
    for method in ["union", "dgcg"]:
        feats = L.load_feats(name)
        splits = split_classes(sorted(feats), 5, [10, 5, 5], 7)
        a = L.make_args(["--class-split", "10", "5", "5", "--device", L.DEVICE, *L.METHODS[method][2]])
        model = L.build_model(a, 512, L.METHODS[method][1], "proto-sgc")
        for shot in [5]:
            sampler = FeatureEpisodeSampler(feats, 5, shot, 10)
            rng = np.random.default_rng(999)
            ep_sup, ep_q = [], []
            with torch.no_grad():
                for _ in range(150):
                    ep = sampler.sample(splits["test"], rng)
                    X = torch.cat((ep.support_x, ep.query_x)); y = torch.cat((ep.support_y, ep.query_y))
                    A = model.graph_adjacency(X).clone(); A.fill_diagonal_(0)
                    same = (y[:, None] == y[None, :]).float(); ns = len(ep.support_y)
                    S = A[:ns, :ns]
                    if S.sum() > 0: ep_sup.append(float((S * same[:ns, :ns]).sum() / S.sum()))
                    else: ep_sup.append(np.nan)
                    Q = A[ns:]; ep_q.append(float((Q * same[ns:]).sum() / Q.sum()))
            es, eq = np.array(ep_sup), np.array(ep_q); ok = ~np.isnan(es)
            out.append(dict(dataset=name, method=method, shot=shot, sup_hom=100*np.nanmean(es), q_hom=100*eq.mean(),
                            ep_corr=float(np.corrcoef(es[ok], eq[ok])[0, 1]), frac_sem_aresta_sup=float(1-ok.mean())))
            print(out[-1], flush=True)
json.dump(out, open("res_supproxy.json", "w"))
import pandas as pd
d = pd.DataFrame(out)
for m in ["union", "dgcg"]:
    s = d[d.method == m]; print(m, "corr entre bases (sup_hom x q_hom):", round(s[["sup_hom", "q_hom"]].corr().iloc[0, 1], 3), "corr média por episódio:", round(s.ep_corr.mean(), 3))
