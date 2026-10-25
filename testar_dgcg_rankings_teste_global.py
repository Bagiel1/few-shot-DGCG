#!/usr/bin/env python3
"""Diagnostico opt-in: rankings de toda a particao de teste, sem retreinar.

Compara as duas avaliacoes no mesmo checkpoint e nos mesmos episodios.
As referencias globais contêm somente embeddings de teste, sem rótulos nas
correlações. O grafo/propagacao continuam episodicos. Nenhum default do pipeline
principal e modificado. Este protocolo usa transducao sobre a particao inteira.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from proto_sgc.episodes import FeatureEpisodeSampler, episode_to_device
from proto_sgc.graphs.training_rankings import TrainingRankings
from proto_sgc.model import ProtoSGC
from proto_sgc.runtime import _safe_torch_load, seed_everything, select_device
from proto_sgc.training import Metrics


def summarize(losses: list[float], accuracies: list[float]) -> Metrics:
    values = np.asarray(accuracies, dtype=np.float64)
    return Metrics(
        loss=float(np.mean(losses)), accuracy=float(values.mean()),
        ci95=float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values)>1 else 0.,
    )


@torch.no_grad()
def compare(model, sampler, test_classes, reference, episodes, seed, device):
    model.eval()
    rng = np.random.default_rng(seed)
    losses = {'episodico': [], 'teste_global': []}
    accuracies = {'episodico': [], 'teste_global': []}
    detail = []
    for index in range(episodes):
        episode = episode_to_device(sampler.sample(test_classes, rng), device)
        correlations = reference.correlations(
            episode.sample_ids, correlation=model.dgcg_correlation,
            top_k=model.dgcg_top_k, rbo_p=model.dgcg_rbo_p,
            dtype=model.theta.weight.dtype, device=device,
        )
        row = {'episodio': index+1}
        for mode in losses:
            options = {'dgcg_correlations':correlations} if mode=='teste_global' else {}
            logits = model(episode.support_x, episode.support_y, episode.query_x,
                           sampler.n_way, **options)
            loss = float(F.cross_entropy(logits, episode.query_y).item())
            accuracy = float((logits.argmax(1)==episode.query_y).float().mean().item())
            losses[mode].append(loss)
            accuracies[mode].append(accuracy)
            row[mode+'_accuracy'] = accuracy
            row[mode+'_loss'] = loss
        detail.append(row)
        if (index+1)%200==0 or index+1==episodes:
            print(f'Avaliacao pareada: {index+1}/{episodes} episodios', flush=True)
    return {mode:summarize(losses[mode],accuracies[mode]) for mode in losses}, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--features-cache',type=Path)
    parser.add_argument('--episodes',type=int,help='Default: test_episodes do checkpoint.')
    parser.add_argument('--device',choices=('auto','cpu','cuda','mps'),default='auto')
    parser.add_argument('--output',type=Path)
    options = parser.parse_args()
    # Checkpoint local produzido por este projeto: args inclui objetos Path.
    checkpoint = torch.load(options.checkpoint,map_location='cpu',weights_only=False)
    args = checkpoint['args']
    if args['model']!='proto-sgc' or args['graph_type'] not in ('dgcg','dgcg-plus'):
        parser.error('O checkpoint precisa ser Proto-SGC com DGCG/DGCG+.')
    count = options.episodes if options.episodes is not None else args['test_episodes']
    if count<1:parser.error('--episodes deve ser >= 1.')
    seed_everything(args['seed'])
    device = select_device(options.device)
    cache_path = options.features_cache or options.checkpoint.parent / (
        f"features_resnet18_{args['weights']}_{args['image_size']}px.pt"
    )
    cached = _safe_torch_load(cache_path)
    features = cached['features_by_class']
    if int(cached['input_dim'])!=checkpoint['input_dim']:
        parser.error('Cache e checkpoint têm dimensoes diferentes.')
    splits = checkpoint['class_splits']
    if any(set(splits[a])&set(splits[b]) for a,b in [('train','val'),('train','test'),('val','test')]):
        parser.error('Splits de classes devem ser disjuntos.')
    if set(features)!=set(splits['train']+splits['val']+splits['test']):
        parser.error('Classes do cache diferem das classes do checkpoint.')
    model_keys = ['model_name','knn','graph_type','sgc_hops','graph_temperature','grande',
                  'grande_sigma','grande_metric','grande_rbo_p','cosine_rbf_weight',
                  'dgcg_metric','dgcg_correlation','dgcg_list_size','dgcg_candidate_k',
                  'dgcg_top_k','dgcg_threshold','dgcg_target_degree','dgcg_target_density',
                  'dgcg_rbo_p','knn_metric']
    kwargs = {key:args[key] for key in model_keys if key in args}
    kwargs['model_name'] = args['model']
    model = ProtoSGC(input_dim=checkpoint['input_dim'],output_dim=args['feature_dim'],**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    sampler = FeatureEpisodeSampler(features,args['n_way'],args['n_shot'],args['n_query'])
    # Banco restrito ao teste. Nao concatena imagens de treino ou validacao.
    reference = TrainingRankings(features,splits['test'],list_size=args['dgcg_list_size'],
                                metric=args['dgcg_metric'])
    print(f'Checkpoint fixo: {options.checkpoint}')
    print(f"Protocolo: {args['n_way']}-way/{args['n_shot']}-shot/{args['n_query']}-query; device={device}")
    print(f'Referencias somente de teste: {len(reference.features)} imagens; L={reference.rankings.shape[1]}')
    print(f"Correlacao: {args['dgcg_correlation']}, top-k={args['dgcg_top_k']}; seed episodica={args['seed']+303}")
    metrics, detail = compare(model,sampler,splits['test'],reference,count,args['seed']+303,device)
    for mode,m in metrics.items():
        print(f'{mode}: {100*m.accuracy:.2f}% +/- {100*m.ci95:.2f}% (IC 95%) | loss={m.loss:.4f}')
    delta = 100*(metrics['teste_global'].accuracy-metrics['episodico'].accuracy)
    print(f'Diferenca teste global - episodico: {delta:+.2f} pp')
    output = options.output or options.checkpoint.parent/'comparacao_rankings_teste_global.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({
        'checkpoint':str(options.checkpoint.resolve()),
        'checkpoint_sha256':hashlib.sha256(options.checkpoint.read_bytes()).hexdigest(),
        'features_cache':str(cache_path.resolve()),
        'n_way':args['n_way'],'n_shot':args['n_shot'],'n_query':args['n_query'],
        'episodes':count,'seed':args['seed']+303,'device':str(device),
        'reference_classes':splits['test'],'reference_images':len(reference.features),
        'list_size':args['dgcg_list_size'],'top_k':args['dgcg_top_k'],
        'protocol':'Mesmo checkpoint; referencias de toda a particao de teste sem rotulos; grafo episodico.',
        'metrics':{k:asdict(v) for k,v in metrics.items()},'difference_pp':delta,
        'episode_results':detail,
    },ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Resultados: {output}')


if __name__=='__main__':main()
