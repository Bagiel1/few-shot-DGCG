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


class QueryContextBank:
    """Separa queries avaliadas e contexto nao rotulado de um episodio mestre."""

    def __init__(self, sampler: FeatureEpisodeSampler, classes: Sequence[str], *,
                 pool_per_class: int, active_per_class: int):
        if pool_per_class < 0 or not 0 <= active_per_class <= pool_per_class:
            raise ValueError("O contexto ativo deve estar entre zero e o tamanho do pool.")
        self.classes = sorted(classes)
        self.pool_per_class = pool_per_class
        self.active_per_class = active_per_class
        self.count = sampler.n_way * active_per_class
        self.n_way = sampler.n_way
        self.n_shot = sampler.n_shot
        self.n_query = sampler.n_query
        self.master_query_count = sampler.n_query + pool_per_class
        self.master_sampler = FeatureEpisodeSampler(
            sampler.features_by_class,
            n_way=sampler.n_way,
            n_shot=sampler.n_shot,
            n_query=self.master_query_count,
        )
        self._last_episode_ids: tuple[tuple[str, int], ...] | None = None
        self._last_extra_x: torch.Tensor | None = None

    def sample(self, rng: np.random.Generator) -> Episode:
        master = self.master_sampler.sample(self.classes, rng)
        n_support = self.n_way * self.n_shot
        master_query_ids = master.sample_ids[n_support:]
        evaluated_positions: list[int] = []
        context_positions: list[int] = []
        for class_index in range(self.n_way):
            start = class_index * self.master_query_count
            evaluated_positions.extend(range(start, start + self.n_query))
            context_start = start + self.n_query
            context_positions.extend(
                range(context_start, context_start + self.active_per_class)
            )

        evaluated_index = torch.tensor(evaluated_positions, dtype=torch.long)
        context_index = torch.tensor(context_positions, dtype=torch.long)
        evaluated_ids = tuple(master_query_ids[index] for index in evaluated_positions)
        episode = Episode(
            support_x=master.support_x,
            support_y=master.support_y,
            query_x=master.query_x[evaluated_index],
            query_y=master.query_y[evaluated_index],
            sample_ids=master.sample_ids[:n_support] + evaluated_ids,
        )
        self._last_episode_ids = episode.sample_ids
        self._last_extra_x = master.query_x[context_index]
        return episode

    def extra_features(self, episode: Episode) -> torch.Tensor:
        if episode.sample_ids != self._last_episode_ids or self._last_extra_x is None:
            raise ValueError("O contexto solicitado nao pertence ao ultimo episodio amostrado.")
        return self._last_extra_x

    def metadata(self) -> dict:
        return {
            "kind": "queries_from_same_master_episode",
            "pool_per_class": self.pool_per_class,
            "active_per_class": self.active_per_class,
            "evaluated_queries_per_class": self.n_query,
            "selection": "first_queries_evaluated; following_queries_are_nested_context",
            "count": self.count,
        }
