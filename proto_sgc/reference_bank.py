"""Banco reservado de imagens nao rotuladas das classes de um episodio."""

from collections.abc import Sequence

import numpy as np
import torch

from .episodes import Episode, FeatureEpisodeSampler


class ExtraReferenceBank:
    def __init__(self, sampler: FeatureEpisodeSampler, classes: Sequence[str], *,
                 count: int, per_class: int, seed: int):
        if count < 0 or per_class < 1 or count > sampler.n_way * per_class:
            raise ValueError("Quantidade de extras/banco invalida para n-way.")
        self.count, self.per_class, self.seed = count, per_class, seed
        self.classes = sorted(classes)
        generator = np.random.default_rng(seed)
        self.bank: dict[str, torch.Tensor] = {}
        self.bank_ids: dict[str, list[int]] = {}
        self.episode_ids: dict[str, list[int]] = {}
        remaining = {}
        required = per_class + sampler.n_shot + sampler.n_query
        for label in self.classes:
            features = sampler.features_by_class[label]
            if len(features) < required:
                raise ValueError(
                    f"Classe {label!r}: {len(features)} imagens, mas banco+shot+query "
                    f"requerem {required}. Reduza --dgcg-extra-bank-per-class."
                )
            indices = generator.permutation(len(features))
            self.bank_ids[label] = indices[:per_class].tolist()
            self.episode_ids[label] = indices[per_class:].tolist()
            self.bank[label] = features[torch.as_tensor(indices[:per_class])]
            remaining[label] = features[torch.as_tensor(indices[per_class:])]
        self.sampler = FeatureEpisodeSampler(remaining, sampler.n_way, sampler.n_shot, sampler.n_query)

    def sample(self, rng: np.random.Generator) -> Episode:
        return self.sampler.sample(self.classes, rng)

    def extra_features(self, episode: Episode) -> torch.Tensor:
        # Classes servem somente a amostragem do protocolo: a correlacao recebe
        # exclusivamente os vetores. Apenas classes presentes entram no banco.
        labels = list(dict.fromkeys(label for label, _ in episode.sample_ids))
        if len(labels) != self.sampler.n_way:
            raise ValueError("Identificadores do episodio incompatíveis com o banco.")
        quotient, remainder = divmod(self.count, len(labels))
        blocks = [self.bank[label][:quotient + (index < remainder)]
                  for index, label in enumerate(labels)]
        return torch.cat(blocks)

    def metadata(self) -> dict:
        return {"count":self.count, "per_class":self.per_class, "seed":self.seed,
                "bank_ids":self.bank_ids, "episode_ids":self.episode_ids}
