import sys, json, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
df = pd.concat([pd.read_json(p, lines=True) for p in sys.argv[1:]])
df["cfg"] = df.backbone + "|" + df.feat + "|" + df.trainfree.map({True: "tf", False: "tr"}) + df.tag.fillna("")
short = {n: n.replace("_micro", "")[:6] for n in df.dataset.unique()}
df["ds"] = df.dataset.map(short)
for shot in sorted(df.shot.unique()):
    s = df[df.shot == shot]
    print(f"\n===== 5-way {shot}-shot: média TESTE nas bases (n bases) =====")
    t = s.pivot_table(index="cfg", columns="method", values="test", aggfunc="mean").round(2)
    n = s.pivot_table(index="cfg", columns="method", values="test", aggfunc="count")
    v = s.pivot_table(index="cfg", columns="method", values="val", aggfunc="mean").round(2)
    print(t.join(v, rsuffix="_val").join(n.min(axis=1).rename("n")))
    print(f"\n-- por base (teste), {shot}-shot --")
    print(s.pivot_table(index=["cfg"], columns=["ds", "method"], values="test", aggfunc="mean").round(1).T.unstack(level=1) if len(sys.argv) > 9 else
          s.pivot_table(index=["ds", "method"], columns="cfg", values="test", aggfunc="mean").round(1))
