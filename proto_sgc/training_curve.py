"""Diagnóstico pareado sobre os grafos usados durante o meta-treino."""

import hashlib
import json
from collections import defaultdict
from statistics import mean

import torch

from .homophily import edge_agreement, pearson
from .training import evaluate, train_model


def train_with_curve(model, sampler, splits, args, device):
    episodes, checkpoints = [], []
    adjacency = None

    def capture(graph):
        nonlocal adjacency
        adjacency = graph.detach()

    def record_episode(step, episode, accuracy):
        row = dict(episode=step, train_percent=100 * accuracy,
                   sample_hash=hashlib.sha256(json.dumps(episode.sample_ids).encode()).hexdigest())
        if model.model_name == "proto-sgc":
            labels = torch.cat((episode.support_y, episode.query_y))
            row.update(edge_agreement(adjacency, labels, len(episode.support_x)))
        episodes.append(row)

    def record_checkpoint(step, validation):
        metrics = evaluate(model, sampler, splits["test"], args.test_episodes,
                           args.n_way, device, seed=args.seed + 303)
        checkpoints.append(dict(episode=step, validation_percent=100 * validation.accuracy,
                                test_percent=100 * metrics.accuracy))

    previous = model.graph_observer
    model.graph_observer = capture
    try:
        model, metrics, path = train_model(
            model, sampler, splits, args, device,
            on_episode=record_episode, on_validation=record_checkpoint,
        )
    finally:
        model.graph_observer = previous
    return model, metrics, path, dict(episodes=episodes, checkpoints=checkpoints)


def paired_curves(root, jobs):
    runs = {}
    for job in jobs:
        path = root / job["folder"] / "diagnostic.json"
        if path.exists():
            row = json.loads(path.read_text())
            runs[row["dataset"], row["shot"], row["seed"], row["method"]] = row

    paired, windows, summary = [], [], []
    by_scope = defaultdict(list)
    for (dataset, shot, seed, method), graph in runs.items():
        baseline = runs.get((dataset, shot, seed, "no-graph"))
        if method == "no-graph" or baseline is None:
            continue
        identity = dict(dataset=dataset, shot=shot, seed=seed, method=method)
        graph_trace, baseline_trace = graph["curve"], baseline["curve"]
        current = []
        for a, b in zip(graph_trace["episodes"], baseline_trace["episodes"], strict=True):
            if (a["episode"], a["sample_hash"]) != (b["episode"], b["sample_hash"]):
                raise ValueError(f"Episódios não pareados: {identity}")
            row = dict(identity, **a, baseline_train_percent=b["train_percent"],
                       gain_train_pp=a["train_percent"] - b["train_percent"],
                       homophily_query_percent=100*a["same_query"]/a["weight_query"]
                       if a["weight_query"] else None)
            current.append(row)
        paired.extend(current)

        blocks = []
        start = 0
        for a, b in zip(graph_trace["checkpoints"], baseline_trace["checkpoints"], strict=True):
            stop = a["episode"]
            if stop != b["episode"]:
                raise ValueError(f"Checkpoints não pareados: {identity}")
            chunk = current[start:stop]
            weight = sum(r["weight_query"] for r in chunk)
            blocks.append(dict(identity, first_episode=start+1, episode=stop,
                               homophily_query_percent=100*sum(r["same_query"] for r in chunk)/weight if weight else None,
                               train_percent=mean(r["train_percent"] for r in chunk),
                               baseline_train_percent=mean(r["baseline_train_percent"] for r in chunk),
                               gain_train_pp=mean(r["gain_train_pp"] for r in chunk),
                               test_percent=a["test_percent"], baseline_test_percent=b["test_percent"],
                               gain_test_pp=a["test_percent"]-b["test_percent"]))
            start = stop
        windows.extend(blocks)
        first_train = next((r for r in blocks if r["gain_train_pp"] > 0), {})
        first_test = next((r for r in blocks if r["gain_test_pp"] > 0), {})
        valid = [r for r in current if r["homophily_query_percent"] is not None]
        valid_blocks = [r for r in blocks if r["homophily_query_percent"] is not None]
        weight = sum(r["weight_query"] for r in current)
        result = dict(identity, first_train_lead_start=first_train.get("first_episode"),
                      first_train_lead_end=first_train.get("episode"),
                      first_test_lead_checkpoint=first_test.get("episode"),
                      pearson_episode_train=pearson([r["homophily_query_percent"] for r in valid],
                                                    [r["gain_train_pp"] for r in valid]),
                      pearson_window_train=pearson([r["homophily_query_percent"] for r in valid_blocks],
                                                   [r["gain_train_pp"] for r in valid_blocks]),
                      pearson_window_test=pearson([r["homophily_query_percent"] for r in valid_blocks],
                                                  [r["gain_test_pp"] for r in valid_blocks]),
                      train_homophily_query_percent=100*sum(r["same_query"] for r in current)/weight if weight else None,
                      gain_final_test_pp=graph["test_percent"]-baseline["test_percent"])
        summary.append(result)
        if weight:
            by_scope[shot, seed, method].append(result)
            by_scope[shot, seed, "both_graphs"].append(result)

    across_bases = [
        dict(shot=shot, seed=seed, method=method, points=len(rows),
             bases=len({r["dataset"] for r in rows}),
             pearson_train_homophily_final_test_gain=pearson(
                 [r["train_homophily_query_percent"] for r in rows],
                 [r["gain_final_test_pp"] for r in rows]))
        for (shot, seed, method), rows in by_scope.items()
    ]
    return paired, windows, summary, across_bases
