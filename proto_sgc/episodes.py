"""Divisão por classes e amostragem dos conjuntos de suporte e consulta de cada episódio."""

from __future__ import annotations

from dataclasses import dataclass
from _collections_abc import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class Episode:
    """Tensores de suporte e consulta que compoem um episodio few-shot.

    Os rotulos armazenados aqui sao locais ao episodio: sempre variam de
    ``0`` a ``n_way - 1``, independentemente do nome real da classe.
    """

    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor


def split_classes(
    class_names: Sequence[str],
    n_way: int,
    requested_split: Sequence[int] | None,
    seed: int,
) -> dict[str, list[str]]:
    """Divide classes, e nao imagens, entre meta-treino, validacao e teste.

    Essa separacao por classe e o que caracteriza a avaliacao em classes nao
    vistas: nenhuma classe de validacao ou teste participa do meta-treino.
    """

    n_classes = len(class_names)
    if requested_split is None:
        # Por padrao, validacao e teste recebem n_way classes cada; todas as
        # classes restantes ficam no meta-treino.
        n_val = n_way
        n_test = n_way
        n_train = n_classes - n_val - n_test
        split_sizes = (n_train, n_val, n_test)
    else:
        split_sizes = tuple(int(value) for value in requested_split)

    if sum(split_sizes) != n_classes:
        raise ValueError(
            f"--class-split soma {sum(split_sizes)}, mas o dataset tem {n_classes} classes."
        )
    if any(size < n_way for size in split_sizes):
        raise ValueError(
            f"Cada particao precisa ter ao menos n-way={n_way} classes; split={split_sizes}."
        )

    # A ordenacao antes da permutacao impede que a ordem fornecida pelo sistema
    # de arquivos altere a divisao obtida com a mesma semente.
    generator = np.random.default_rng(seed)
    shuffled = list(generator.permutation(sorted(class_names)))
    n_train, n_val, _ = split_sizes
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


class FeatureEpisodeSampler:
    """Amostra episodios N-way/K-shot a partir de embeddings pre-calculados."""

    def __init__(
        self,
        features_by_class: dict[str, torch.Tensor],
        n_way: int,
        n_shot: int,
        n_query: int,
    ) -> None:
        """Configura o protocolo e verifica se todas as classes têm exemplos."""

        self.features_by_class = features_by_class
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        required = n_shot + n_query
        too_small = {
            label: len(features)
            for label, features in features_by_class.items()
            if len(features) < required
        }
        if too_small:
            raise ValueError(
                f"Cada classe precisa de {required} imagens; insuficientes: {too_small}"
            )

    def sample(self, allowed_classes: Sequence[str], rng: np.random.Generator) -> Episode:
        """Cria um episodio sem reposicao dentro do conjunto de classes permitido.

        Primeiro escolhe ``n_way`` classes. Para cada uma, reserva ``n_shot``
        vetores ao suporte e ``n_query`` vetores à consulta. Os rotulos reais
        sao remapeados para ``0, ..., n_way-1`` apenas dentro deste episodio.
        """

        # Nao ha repeticao de classe dentro do episodio.
        selected = list(rng.choice(allowed_classes, size=self.n_way, replace=False))
        support_x: list[torch.Tensor] = []
        query_x: list[torch.Tensor] = []
        support_y: list[int] = []
        query_y: list[int] = []

        for episodic_label, class_name in enumerate(selected):
            class_features = self.features_by_class[str(class_name)]

            # Suporte e consulta sao disjuntos porque a amostragem e sem reposicao.
            indices = rng.choice(
                len(class_features),
                size=self.n_shot + self.n_query,
                replace=False,
            )
            support_indices = torch.as_tensor(indices[: self.n_shot], dtype=torch.long)
            query_indices = torch.as_tensor(indices[self.n_shot :], dtype=torch.long)
            support_x.append(class_features[support_indices])
            query_x.append(class_features[query_indices])
            support_y.extend([episodic_label] * self.n_shot)
            query_y.extend([episodic_label] * self.n_query)

        # A concatenacao coloca os exemplos classe a classe. O modelo nao
        # depende dessa ordem, pois seleciona o suporte por seu rotulo local.
        return Episode(
            support_x=torch.cat(support_x, dim=0),
            support_y=torch.tensor(support_y, dtype=torch.long),
            query_x=torch.cat(query_x, dim=0),
            query_y=torch.tensor(query_y, dtype=torch.long),
        )


def episode_to_device(episode: Episode, device: torch.device) -> Episode:
    """Move os quatro tensores de um episodio para o dispositivo de execucao."""

    return Episode(
        support_x=episode.support_x.to(device),
        support_y=episode.support_y.to(device),
        query_x=episode.query_x.to(device),
        query_y=episode.query_y.to(device),
    )
