import sys, json, itertools, traceback
from multiprocessing import get_context
import fs_lab as L

def grid(name):
    if name == "e1":  # pré-processamento × treino × método
        for d, shot, m, feat, tf in itertools.product(L.NAMES, [1, 5], ["no-graph", "union", "dgcg"],
                                                       ["raw", "cl2n", "pt-cl2n"], [False, True]):
            yield dict(name=d, method=m, shot=shot, feat=feat, trainfree=tf)
    elif name == "idinit":  # treino partindo da identidade, 512-d
        for d, shot, m in itertools.product(L.NAMES, [1, 5], ["no-graph", "union", "dgcg24"]):
            yield dict(name=d, method=m, shot=shot, feat="raw", trainfree=False, tag="idinit", extra=["--feature-dim", "512"])
    elif name.startswith("proptf:"):  # propagação sem treino (barato)
        feat = name.split(":")[1]
        tags = {"hops1": ["--sgc-hops", "1"], "hops3": ["--sgc-hops", "3"], "hops4": ["--sgc-hops", "4"],
                "alpha0.5": [], "hops4alpha0.5": ["--sgc-hops", "4"]}
        for d, shot, m, (tag, ex) in itertools.product(L.NAMES, [1, 5], ["union", "dgcg24"], tags.items()):
            yield dict(name=d, method=m, shot=shot, feat=feat, trainfree=True, tag=tag, extra=ex)
    elif name.startswith("topo"):  # topo:tf (sem treino) | topo:tr:<lista de variantes>
        parts = name.split(":"); tf = parts[1] == "tf"
        var = [
            ("union", "", []), ("union", "topo:sup", []), ("union", "topo:snn", []),
            ("union", "cosrbf", ["--cosine-rbf-weight"]), ("union", "grande", ["--grande"]),
            ("dgcg", "", []), ("dgcg", "topo:wrbo", []), ("dgcg", "topo:wrbo2", []), ("dgcg", "topo:snn", []),
            ("dgcg", "topo:sym", []), ("dgcg", "topo:mut", []), ("dgcg", "topo:sup", []),
            ("dgcg", "topo:supcut", []), ("dgcg", "topo:prune", []), ("dgcgplus", "", []),
            ("dgcg", "cosrbf", ["--cosine-rbf-weight"]), ("dgcg", "grande", ["--grande"]),
            ("no-graph", "", []),
        ]
        if not tf:
            keep = parts[2].split(",")
            var = [v for v in var if f"{v[0]}|{v[1]}" in keep]
        for d, shot, (m, tag, ex) in itertools.product(L.NAMES, [1, 5], var):
            yield dict(name=d, method=m, shot=shot, feat="raw", trainfree=tf, tag=tag, extra=ex)
    elif name == "dgcg24":
        for d, shot, feat, tf in itertools.product(L.NAMES, [1, 5], ["raw", "cl2n"], [False, True]):
            yield dict(name=d, method="dgcg24", shot=shot, feat=feat, trainfree=tf)
    elif name.startswith("bb:"):  # backbone: bb:resnet50:224
        _, bb, size = name.split(":")
        for d, shot, m, feat, tf in itertools.product(L.NAMES, [1, 5], ["no-graph", "union", "dgcg"],
                                                       ["raw", "cl2n"], [False, True]):
            yield dict(name=d, method=m, shot=shot, feat=feat, trainfree=tf, backbone=bb, size=int(size))
    elif name.startswith("prop:"):  # força da propagação; prop:<feat>
        feat = name.split(":")[1]
        tags = {"hops1": ["--sgc-hops", "1"], "hops3": ["--sgc-hops", "3"], "hops4": ["--sgc-hops", "4"],
                "alpha0.3": [], "alpha0.6": []}
        for d, shot, m, (tag, ex) in itertools.product(L.NAMES, [1, 5], ["union", "dgcg"], tags.items()):
            yield dict(name=d, method=m, shot=shot, feat=feat, trainfree=False, tag=tag, extra=ex)
    elif name == "seeds":
        for d, shot, m, seed in itertools.product(L.NAMES, [1, 5], ["no-graph", "union", "dgcg"], [17, 27]):
            for feat in ["raw", "cl2n"]:
                yield dict(name=d, method=m, shot=shot, feat=feat, trainfree=False, seed=seed)

def key(j):
    return (j["name"], j["method"], 5, j["shot"], j.get("seed", 7), j["feat"],
            f'{j.get("backbone","resnet18")}@{j.get("size",128)}', j["trainfree"], j.get("tag", ""))

def work(j):
    try:
        return L.run(**j)
    except Exception:
        return {"error": traceback.format_exc(), "job": j}

if __name__ == "__main__":
    g, out, nproc = sys.argv[1], sys.argv[2], int(sys.argv[3])
    done = L.done_keys(out)
    jobs = [j for j in grid(g) if key(j) not in done]
    jobs.sort(key=lambda j: j["trainfree"])  # treinos primeiro
    print(f"{len(jobs)} jobs", flush=True)
    with get_context("spawn").Pool(nproc) as p:
        for i, r in enumerate(p.imap_unordered(work, jobs), 1):
            if "error" in r:
                print("ERRO", r["job"], r["error"][-500:], flush=True)
                with open(out + ".err", "a") as f: f.write(json.dumps(r) + "\n")
            else:
                L.append(out, r)
            if i % 20 == 0: print(f"{i}/{len(jobs)}", flush=True)
    print("FIM", flush=True)
