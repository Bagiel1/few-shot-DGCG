#!/usr/bin/env python3
"""Entrada do experimento ProtoNet + SGC.

Execute como antes: ``python protosgc_fewshot.py --quick``.
O fluxo está em ``proto_sgc/experiment.py``; consulte ``documentacao.md``
para o mapa dos arquivos e a ordem sugerida de leitura.

As importações abaixo preservam o acesso às funções e classes que antes
estavam neste arquivo, inclusive para notebooks que importam ProtoSGC.
"""

from proto_sgc.config import (
    FLOWERS_MICRO_OPENML_ID as FLOWERS_MICRO_OPENML_ID,
    IMAGE_EXTENSIONS as IMAGE_EXTENSIONS,
    DGCG_GRAPH_TYPES as DGCG_GRAPH_TYPES,
    GRAPH_TYPES as GRAPH_TYPES,
    DGCG_CORRELATIONS as DGCG_CORRELATIONS,
    DGCG_METRICS as DGCG_METRICS,
    GRANDE_METRICS as GRANDE_METRICS,
    parse_args as parse_args,
    validate_hyperparameters as validate_hyperparameters,
)
from proto_sgc.data import (
    ImageRecord as ImageRecord,
    ImageRecordDataset as ImageRecordDataset,
    _case_insensitive_column as _case_insensitive_column,
    _resolve_image_records as _resolve_image_records,
    load_local_meta_album as load_local_meta_album,
    download_meta_album as download_meta_album,
)
from proto_sgc.features import (
    records_fingerprint as records_fingerprint,
    build_resnet18 as build_resnet18,
    image_transform as image_transform,
    extract_or_load_features as extract_or_load_features,
)
from proto_sgc.episodes import (
    Episode as Episode,
    FeatureEpisodeSampler as FeatureEpisodeSampler,
    split_classes as split_classes,
    episode_to_device as episode_to_device,
)
from proto_sgc.model import (
    ProtoSGC as ProtoSGC,
)
from proto_sgc.runtime import (
    seed_everything as seed_everything,
    select_device as select_device,
    _safe_torch_load as _safe_torch_load,
)
from proto_sgc.training import (
    Metrics as Metrics,
    evaluate as evaluate,
    train_model as train_model,
)
from proto_sgc.experiment import (
    main as main,
)


if __name__ == "__main__":
    main()
