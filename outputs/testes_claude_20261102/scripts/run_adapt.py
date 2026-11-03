import itertools, json, fs_lab as L
from multiprocessing import get_context
def w(j):
    return L.run(**j)
if __name__ == "__main__":
    jobs = [dict(name=d, method=m, shot=5, trainfree=True, tag=t)
            for d, m, t in itertools.product(L.NAMES, ["union", "dgcg"], ["adapt", "adapt0.45-0.85"])]
    with get_context("spawn").Pool(1) as p:
        for r in p.imap_unordered(w, jobs):
            L.append("res_adapt.jsonl", r)
    print("FIM")
