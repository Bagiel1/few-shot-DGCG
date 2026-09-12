"""Leitura do Meta-Album local ou via OpenML e adaptação das imagens para o PyTorch."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class ImageRecord:
    """Caminho de uma imagem e nome textual de sua classe original."""

    path: Path
    label: str


def _case_insensitive_column(columns: Sequence[object], wanted: str) -> str:
    """Localiza uma coluna ignorando maiusculas e espacos nas extremidades."""

    mapping = {str(column).strip().upper(): str(column) for column in columns}
    try:
        return mapping[wanted.upper()]
    except KeyError as error:
        raise ValueError(
            f"Coluna {wanted!r} nao encontrada. Colunas disponiveis: {list(columns)}"
        ) from error


def _resolve_image_records(
    root: Path,
    file_names: Sequence[object],
    labels: Sequence[object],
) -> list[ImageRecord]:
    """Resolve os nomes do metadado para arquivos de imagem existentes.

    A estrutura de pastas pode variar entre uma copia manual do Meta-Album e
    diferentes versoes do cliente OpenML. Por isso, primeiro sao testados os
    caminhos usuais e, como ultimo recurso, uma correspondencia pelo nome do
    arquivo em toda a arvore.
    """

    root = root.resolve()

    # Este indice evita repetir uma busca recursiva para cada linha do CSV.
    all_images = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images_by_name: dict[str, list[Path]] = defaultdict(list)
    for path in all_images:
        images_by_name[path.name].append(path)

    records: list[ImageRecord] = []
    missing: list[str] = []
    for raw_name, raw_label in zip(file_names, labels, strict=True):
        relative = Path(str(raw_name))

        # Meta-Album normalmente armazena as imagens em images/, enquanto
        # alguns pacotes do OpenML preservam o caminho relativo diretamente.
        candidates = (
            root / "images" / relative,
            root / relative,
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        # A busca pelo basename so e segura quando existe uma unica ocorrencia.
        if path is None:
            matches = images_by_name.get(relative.name, [])
            path = matches[0] if len(matches) == 1 else None
        if path is None:
            missing.append(str(raw_name))
        else:
            records.append(ImageRecord(path=path.resolve(), label=str(raw_label)))

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Nao foi possivel localizar {len(missing)} imagens em {root}. "
            f"Exemplos: {preview}. Atualize o pacote openml e confirme que "
            "download_all_files=True conseguiu baixar os arquivos auxiliares. "
            "Como alternativa, baixe o Meta-Album manualmente e use --data-root."
        )
    return records


def load_local_meta_album(data_root: Path) -> list[ImageRecord]:
    """Carrega um Meta-Album local a partir de ``labels.csv`` e das imagens."""

    data_root = data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"--data-root nao existe: {data_root}")

    # Aceita tanto a raiz exata do dataset quanto um diretorio pai que o
    # contenha, mas rejeita ambiguidades entre varios labels.csv.
    direct = data_root / "labels.csv"
    candidates = [direct] if direct.is_file() else list(data_root.rglob("labels.csv"))
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Esperava exatamente um labels.csv sob {data_root}; encontrei {len(candidates)}."
        )

    labels_path = candidates[0]
    dataset_root = labels_path.parent
    metadata = pd.read_csv(labels_path)
    file_column = _case_insensitive_column(metadata.columns, "FILE_NAME")
    label_column = _case_insensitive_column(metadata.columns, "CATEGORY")
    return _resolve_image_records(
        dataset_root,
        metadata[file_column].tolist(),
        metadata[label_column].tolist(),
    )


def download_meta_album(openml_id: int, cache_dir: Path) -> list[ImageRecord]:
    """Baixa (ou reutiliza do cache) um dataset Meta-Album via OpenML."""

    # O import e tardio para permitir o uso com --data-root mesmo em um
    # ambiente no qual o pacote ``openml`` nao esteja instalado.
    try:
        import openml
    except ImportError as error:
        raise RuntimeError(
            "Instale as dependencias com: pip install -r requirements.txt"
        ) from error

    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(openml.config, "set_root_cache_directory"):
        openml.config.set_root_cache_directory(cache_dir)
    else:  # compatibilidade com openml 0.14
        openml.config.cache_directory = str(cache_dir)

    print(f"Baixando/carregando Meta-Album pelo OpenML (dataset {openml_id})...")
    # ``download_all_files`` e essencial: a tabela contem os nomes, enquanto
    # as imagens propriamente ditas ficam entre os arquivos auxiliares.
    dataset = openml.datasets.get_dataset(
        openml_id,
        download_data=True,
        download_all_files=True,
    )
    target = dataset.default_target_attribute or "CATEGORY"
    frame, target_values, _, _ = dataset.get_data(target=target)
    if target_values is None:
        raise RuntimeError(f"O dataset OpenML {openml_id} nao forneceu a coluna alvo {target!r}.")
    file_column = _case_insensitive_column(frame.columns, "FILE_NAME")

    anchor = getattr(dataset, "parquet_file", None) or getattr(dataset, "data_file", None)
    if anchor is None:
        raise RuntimeError("O cliente OpenML nao informou onde armazenou os dados.")
    dataset_cache_root = Path(anchor).resolve().parent
    records = _resolve_image_records(
        dataset_cache_root,
        frame[file_column].tolist(),
        target_values.tolist(),
    )
    print(f"Dataset: {dataset.name} | imagens: {len(records)}")
    return records


class ImageRecordDataset(Dataset[tuple[torch.Tensor, int]]):
    """Adaptador PyTorch que le imagens e produz ``(tensor, id_da_classe)``."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform: transforms.Compose,
        class_to_id: dict[str, int],
    ) -> None:
        """Armazena os registros, a transformacao e o mapeamento de rotulos."""

        self.records = list(records)
        self.transform = transform
        self.class_to_id = class_to_id

    def __len__(self) -> int:
        """Retorna a quantidade total de imagens."""

        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Abre uma imagem, converte-a para RGB e aplica o pre-processamento."""

        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_id[record.label]
