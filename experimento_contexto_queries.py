#!/usr/bin/env python3
"""Experimento 5-way/1-shot/1-query com 45 queries nao rotuladas no grafo.

O episodio mestre possui 1 suporte e 10 queries por classe. Somente a primeira
query de cada classe entra na loss e nas metricas; as nove restantes participam
apenas da propagacao. Argumentos escritos pelo usuario depois destes padroes
continuam podendo substitui-los.
"""

import sys

from proto_sgc.experiment import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--n-way", "5",
        "--n-shot", "1",
        "--n-query", "1",
        "--extra-source", "queries",
        "--extra-images", "45",
        "--extra-bank-per-class", "9",
        "--extra-mode", "propagation",
        "--extra-phases", "train", "val", "test",
    ]
    main()
