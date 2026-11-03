"""Variantes de topologia/qualidade de aresta aplicadas por wrapper (sem alterar o projeto).

Todas atuam sobre a adjacência ANTES dos autolaços/normalização (graph_adjacency).
Convenção do projeto: A @ X, linha i agrega as colunas j.
"""
import torch
from proto_sgc.graphs import dgcg as G
from proto_sgc.episodes import episode_to_device

def install(model, variant):
    """Instala a variante; o forward guarda n_support/support_y (rótulos do SUPORTE apenas)."""
    if not variant:
        return
    orig_forward, orig_graph = model.forward, model.graph_adjacency
    state = {}

    def forward(support_x, support_y, query_x, n_way, **kw):
        state["ns"], state["sy"] = support_x.shape[0], support_y
        return orig_forward(support_x, support_y, query_x, n_way, **kw)

    def graph(features, **kw):
        A = orig_graph(features, **kw)
        for v in variant.split("+"):
            A = apply(v, A, features, model, state)
        return A
    model.forward, model.graph_adjacency = forward, graph

def rbo_matrix(features, model):
    L = min(model.dgcg_list_size, features.shape[0])
    rl = G.build_ranked_lists(features, L, metric=model.dgcg_metric)
    return G.dgcg_correlation_matrix(rl, correlation="rbo", rbo_p=model.dgcg_rbo_p,
                                     top_k=model.dgcg_top_k, dtype=features.dtype)

def apply(v, A, X, model, st):
    mask = (A != 0).to(A.dtype)
    if v == "sym":   # simetriza (união com transposta)
        return torch.maximum(A, A.T)
    if v == "mut":   # só arestas recíprocas
        return torch.minimum(A, A.T)
    if v == "wrbo":  # peso = correlação RBO da aresta (qualidade pelo contexto de ranking)
        C = rbo_matrix(X, model)
        return mask * C
    if v == "wrbo2": # peso = RBO^2 (mais contraste)
        C = rbo_matrix(X, model)
        return mask * C.square()
    if v == "snn":   # peso = fração de vizinhos compartilhados (SNN) entre extremos
        B = mask.clone(); B.fill_diagonal_(1)
        inter = B @ B.T
        deg = B.sum(1)
        w = inter / torch.sqrt(deg[:, None] * deg[None, :]).clamp_min(1e-12)
        return mask * w
    if v in ("sup", "supcut"):  # usa SOMENTE rótulos do suporte (conhecidos no episódio)
        ns, sy = st["ns"], st["sy"].to(A.device)
        same = (sy[:, None] == sy[None, :]).to(A.dtype)
        A = A.clone()
        block = A[:ns, :ns]
        if v == "sup":
            block = same.clone(); block.fill_diagonal_(0)
        else:
            block = block * same
        A[:ns, :ns] = block
        return A
    if v == "prune":  # remove arestas cuja correlação RBO é < mediana das arestas existentes
        C = rbo_matrix(X, model)
        vals = C[mask.bool()]
        if vals.numel() == 0: return A
        keep = (C >= vals.median()).to(A.dtype)
        B = A * keep
        # evita linha sem vizinho: mantém a melhor aresta original
        empty = B.sum(1) == 0
        if empty.any():
            best = (C * mask).argmax(1)
            rows = torch.nonzero(empty).flatten()
            B[rows, best[rows]] = A[rows, best[rows]]
        return B
    raise ValueError(v)

@torch.no_grad()
def homophily(model, sampler, classes, n_way, episodes, seed):
    """Diagnóstico: fração do peso de arestas (sem autolaços) ligando nós da mesma classe.
    Rótulos de consulta usados SÓ para medir, nunca pelo grafo."""
    import numpy as np
    rng = np.random.default_rng(seed)
    num = den = 0.0; sq = sq_den = 0.0
    for _ in range(episodes):
        ep = sampler.sample(classes, rng)
        ep = episode_to_device(ep, next(model.parameters()).device)
        X = torch.cat((ep.support_x, ep.query_x)); y = torch.cat((ep.support_y, ep.query_y))
        try:
            model.forward(ep.support_x, ep.support_y, ep.query_x, n_way)  # popula estado do suporte
        except Exception:
            pass
        A = model.graph_adjacency(X)
        A = A.clone(); A.fill_diagonal_(0)
        same = (y[:, None] == y[None, :]).to(A.dtype)
        num += float((A * same).sum()); den += float(A.sum())
        ns = ep.support_x.shape[0]   # homofilia só das arestas que chegam às consultas
        Aq = A[ns:]; sq += float((Aq * same[ns:]).sum()); sq_den += float(Aq.sum())
    return 100 * num / max(den, 1e-12), 100 * sq / max(sq_den, 1e-12)

def install_adaptive(model, lo=0.35, hi=0.75):
    """Propagação auto-calibrada por episódio: w = clip((h_sup - lo)/(hi - lo), 0, 1),
    h_sup = homofilia das arestas entre SUPORTES (rótulos conhecidos). A_eff = w*A_hat + (1-w)*I.
    Sem arestas entre suportes (ex.: 1-shot) -> mantém o grafo (w=1)."""
    orig_forward, orig_norm = model.forward, model.normalized_graph_adjacency
    st = {}
    def forward(support_x, support_y, query_x, n_way, **kw):
        st["ns"], st["sy"] = support_x.shape[0], support_y
        return orig_forward(support_x, support_y, query_x, n_way, **kw)
    def norm(features, *a, **kw):
        A_hat = orig_norm(features, *a, **kw)
        A = model.graph_adjacency(features).clone(); A.fill_diagonal_(0)
        ns, sy = st["ns"], st["sy"].to(A.device)
        S = A[:ns, :ns]; same = (sy[:, None] == sy[None, :]).to(A.dtype)
        if float(S.sum()) <= 0 or not bool((same.sum() > ns)):
            w = 1.0
        else:
            h = float((S * same).sum() / S.sum())
            w = min(max((h - lo) / (hi - lo), 0.0), 1.0)
        st.setdefault("ws", []).append(w)
        return w * A_hat + (1 - w) * torch.eye(A_hat.shape[0], dtype=A_hat.dtype, device=A_hat.device)
    model.forward, model.normalized_graph_adjacency = forward, norm
    return st
