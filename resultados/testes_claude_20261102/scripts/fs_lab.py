"""Harness experimental (scratchpad) — importa proto_sgc SEM alterar o projeto."""
from __future__ import annotations
import copy, json, os, sys, time, contextlib, io
from pathlib import Path
PROJ = Path(os.environ.get("FS_PROJ", Path(__file__).resolve().parents[3]))
DEVICE = os.environ.get("FS_DEVICE", "cpu")  # na VM: FS_DEVICE=cuda
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from proto_sgc.config import parse_args, validate_hyperparameters
from proto_sgc.episodes import FeatureEpisodeSampler, split_classes
from proto_sgc.model import ProtoSGC
from proto_sgc.runtime import seed_everything, _safe_torch_load
from proto_sgc.training import train_model, evaluate

NAMES = ["bacteria_micro","plankton_micro","flowers_micro","sports_micro","birds_micro",
         "plant_village_micro","textures_micro","manufacturing_micro","cars_micro","resisc_micro"]
IDS = dict(zip(NAMES, range(44237, 44247)))

def data_root(name):
    base = PROJ/f"data/openml/org/openml/www/datasets/{IDS[name]}"
    if not base.exists():  # máquina sem as bases: baixa pelo OpenML (mesmo cache do projeto)
        from proto_sgc.data import download_meta_album
        download_meta_album(IDS[name], PROJ/"data/openml")
    return [p for p in base.iterdir() if p.is_dir() and p.name.endswith("_Micro")][0]

def load_feats(name, backbone="resnet18", size=128):
    cache = PROJ/f"runs/busca_extras_auto/_work/{name}/features_resnet18_imagenet_128px.pt"
    if backbone == "resnet18" and size == 128 and cache.exists():
        c = _safe_torch_load(PROJ/f"runs/busca_extras_auto/_work/{name}/features_resnet18_imagenet_128px.pt")
        return {k: v.float() for k, v in c["features_by_class"].items()}
    c = torch.load(LAB/f"feats/{name}_{backbone}_{size}.pt", weights_only=False)
    return {k: v.float() for k, v in c.items()}

def transform(feats, splits, kind):
    if kind == "raw": return feats
    out = {}
    if kind in ("cl2n", "pt-cl2n"):
        tr = torch.cat([feats[c] for c in splits["train"]])
        if kind == "pt-cl2n": tr = tr.clamp_min(0).sqrt()
        mu = tr.mean(0, keepdim=True)
    for k, v in feats.items():
        if kind == "l2": v = F.normalize(v, dim=1)
        elif kind == "pt": v = F.normalize(v.clamp_min(0).sqrt(), dim=1)
        elif kind == "cl2n": v = F.normalize(v - mu, dim=1)
        elif kind == "pt-cl2n": v = F.normalize(v.clamp_min(0).sqrt() - mu, dim=1)
        else: raise ValueError(kind)
        out[k] = v.contiguous()
    return out

def make_args(extra):
    old = sys.argv
    sys.argv = ["x"] + extra
    try:
        a = parse_args()
    finally:
        sys.argv = old
    validate_hyperparameters(a)
    return a

def build_model(a, input_dim, graph_type, model_name):
    return ProtoSGC(input_dim=input_dim, output_dim=a.feature_dim, model_name=model_name, knn=a.knn,
        knn_metric=a.knn_metric, graph_type=graph_type, sgc_hops=a.sgc_hops, graph_temperature=a.graph_temperature,
        grande=a.grande, grande_sigma=a.grande_sigma, grande_metric=a.grande_metric, grande_rbo_p=a.grande_rbo_p,
        cosine_rbf_weight=a.cosine_rbf_weight, dgcg_metric=a.dgcg_metric, dgcg_correlation=a.dgcg_correlation,
        dgcg_list_size=a.dgcg_list_size, dgcg_candidate_k=a.dgcg_candidate_k, dgcg_top_k=a.dgcg_top_k,
        dgcg_threshold=a.dgcg_threshold, dgcg_target_degree=a.dgcg_target_degree,
        dgcg_target_density=a.dgcg_target_density, dgcg_rbo_p=a.dgcg_rbo_p)

METHODS = {  # nome -> (model, graph_type, flags extras)
    "no-graph": ("protonet", "knn-union", []),
    "union": ("proto-sgc", "knn-union", []),
    "dgcg": ("proto-sgc", "dgcg", ["--dgcg-target-degree", "4", "6"]),
    "dgcg24": ("proto-sgc", "dgcg", ["--dgcg-target-degree", "2", "4"]),
    "dgcgplus": ("proto-sgc", "dgcg-plus", ["--dgcg-target-degree", "4", "6"]),
}

def run(name, method, *, way=5, shot=1, query=10, seed=7, feat="raw", backbone="resnet18", size=128,
        extra=(), trainfree=False, hook=None, tag=""):
    """Treina (ou avalia sem treino) e devolve dict de resultados."""
    model_name, gtype, mflags = METHODS[method]
    wd = LAB/"work"/f"{name}_{method}_{way}w{shot}s{query}q_s{seed}_{backbone}{size}_{feat}_{'tf' if trainfree else 'tr'}{tag}"
    wd.mkdir(parents=True, exist_ok=True)
    a = make_args(["--n-way", str(way), "--n-shot", str(shot), "--n-query", str(query), "--seed", str(seed),
                   "--class-split", "10", "5", "5", "--model", model_name, "--graph-type", gtype,
                   "--device", DEVICE, "--work-dir", str(wd), *mflags, *extra])
    feats0 = load_feats(name, backbone, size)
    seed_everything(seed)
    splits = split_classes(sorted(feats0), way, a.class_split, seed)
    feats = transform(feats0, splits, feat)
    input_dim = next(iter(feats.values())).shape[1]
    sampler = FeatureEpisodeSampler(feats, way, shot, query)
    seed_everything(seed)
    if trainfree:
        a.feature_dim = input_dim
    model = build_model(a, input_dim, gtype, model_name)
    model.to(DEVICE)
    if hook: hook(model)
    topo_variant = tag.split("topo:")[1] if "topo:" in tag else ""
    import topo as T
    T.install(model, topo_variant)
    if "adapt" in tag:
        lo, hi = (0.35, 0.75) if tag.endswith("adapt") else map(float, tag.split("adapt")[1].split("-"))
        T.install_adaptive(model, lo, hi)
    if "idinit" in tag:  # Theta quadrada iniciada na identidade (treino parte do "sem treino")
        assert a.feature_dim == input_dim, "use --feature-dim igual à dimensão de entrada"
        with torch.no_grad(): model.theta.weight.copy_(torch.eye(input_dim, device=model.theta.weight.device))
    if "alpha" in tag:  # propagação preguiçosa: (1-a)A_hat + aI  (só no experimento)
        alpha = float(tag.split("alpha")[1]); orig = model.normalized_graph_adjacency
        def lazy(*args, **kw):
            A = orig(*args, **kw)
            return (1 - alpha) * A + alpha * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        model.normalized_graph_adjacency = lazy
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if trainfree:
            with torch.no_grad():
                model.theta.weight.copy_(torch.eye(input_dim, device=model.theta.weight.device))
            val = evaluate(model, sampler, splits["val"], a.val_episodes, way, torch.device(DEVICE), seed=seed + 202)
            test = evaluate(model, sampler, splits["test"], a.test_episodes, way, torch.device(DEVICE), seed=seed + 303)
        else:
            a.work_dir = wd
            _, test, ck = train_model(model, sampler, splits, a, torch.device(DEVICE))
            val = None
            js = json.loads(ck.with_suffix(".metrics.json").read_text())
            val_acc = js["validation"]["accuracy"]
    hom = (None, None)
    if model_name == "proto-sgc":
        hom = T.homophily(model, sampler, splits["test"], way, 200, seed + 909)
    r = dict(hom_all=hom[0], hom_query=hom[1], dataset=name, method=method, way=way, shot=shot, query=query, seed=seed, feat=feat,
             backbone=f"{backbone}@{size}", trainfree=trainfree, tag=tag,
             val=100*(val.accuracy if trainfree else val_acc), test=100*test.accuracy, ci=100*test.ci95,
             f1=100*test.macro_f1, deg=(test.graph or {}).get("degree_mean"), secs=round(time.time()-t0, 1))
    return r

def append(path, r):
    with open(path, "a") as f: f.write(json.dumps(r) + "\n")

def done_keys(path):
    if not os.path.exists(path): return set()
    return {(d["dataset"], d["method"], d["way"], d["shot"], d["seed"], d["feat"], d["backbone"], d["trainfree"], d["tag"])
            for d in map(json.loads, open(path))}
