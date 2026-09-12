"""ResNet-18 congelada: pré-processamento, extração dos embeddings e cache em disco."""

from __future__ import annotations

import argparse
import hashlib
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from .data import ImageRecord, ImageRecordDataset
from .runtime import _safe_torch_load


def records_fingerprint(records: Sequence[ImageRecord]) -> str:
    """Resume a identidade dos dados para invalidar caches desatualizados.

    O hash considera caminho, rotulo, tamanho e data de modificacao de cada
    imagem. Ele nao le todos os pixels, evitando duplicar o custo de E/S antes
    da extracao de atributos.
    """

    digest = hashlib.sha256()
    for record in records:
        stat = record.path.stat()
        digest.update(str(record.path).encode("utf-8"))
        digest.update(record.label.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def build_resnet18(weights_name: str) -> tuple[nn.Module, int]:
    """Cria uma ResNet-18 congelada e remove sua camada classificadora."""

    weights = ResNet18_Weights.DEFAULT if weights_name == "imagenet" else None
    backbone = models.resnet18(weights=weights)

    # ``in_features`` e a dimensao do vetor entregue à FC original. Trocando a
    # FC por Identity, esse vetor passa a ser a saida do backbone.
    output_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    backbone.requires_grad_(False)
    backbone.eval()
    return backbone, output_dim


def image_transform(image_size: int) -> transforms.Compose:
    """Monta o pre-processamento esperado pelos pesos ImageNet da ResNet."""

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def extract_or_load_features(
    records: Sequence[ImageRecord],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    """Extrai embeddings congelados ou reutiliza um cache ainda valido.

    O retorno agrupa os vetores por nome de classe. Assim, a amostragem de
    episodios posteriores trabalha somente com tensores em memoria, sem abrir
    imagens nem executar a ResNet durante o meta-treino.
    """

    args.work_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.work_dir / (
        f"features_resnet18_{args.weights}_{args.image_size}px.pt"
    )
    fingerprint = records_fingerprint(records)

    # O cache so e reutilizado quando corresponde exatamente ao conjunto atual
    # de registros. --recompute-features permite ignora-lo explicitamente.
    if cache_path.is_file() and not args.recompute_features:
        cached = _safe_torch_load(cache_path)
        if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
            print(f"Usando embeddings em cache: {cache_path}")
            features = cached["features_by_class"]
            input_dim = int(cached["input_dim"])
            return features, input_dim
        print("O cache de embeddings nao corresponde aos arquivos atuais; recalculando.")

    # IDs inteiros sao temporarios e servem apenas para agrupar os vetores
    # extraidos; as particoes posteriores voltam a usar os nomes das classes.
    class_names = sorted({record.label for record in records})
    class_to_id = {name: index for index, name in enumerate(class_names)}
    dataset = ImageRecordDataset(records, image_transform(args.image_size), class_to_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    backbone, input_dim = build_resnet18(args.weights)
    backbone.to(device)

    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    print(f"Extraindo embeddings com ResNet-18 em {device}...")
    # inference_mode economiza memoria porque o backbone esta congelado e nao
    # participara do meta-treino.
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            feature_batches.append(backbone(images).cpu())
            label_batches.append(labels.cpu())

    all_features = torch.cat(feature_batches, dim=0).float()
    all_labels = torch.cat(label_batches, dim=0)
    # Agrupar uma unica vez torna a composicao de cada episodio barata.
    features_by_class = {
        name: all_features[all_labels == class_to_id[name]].contiguous()
        for name in class_names
    }
    torch.save(
        {
            "fingerprint": fingerprint,
            "input_dim": input_dim,
            "class_names": class_names,
            "features_by_class": features_by_class,
        },
        cache_path,
    )
    print(f"Embeddings salvos em: {cache_path}")
    return features_by_class, input_dim
