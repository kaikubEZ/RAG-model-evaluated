"""The evaluation side reads only datasets/<id>/dataset.csv (docs/dataset-contract.md), never the generator's files."""
from pathlib import Path

from .. import csvio, store
from ..schema import Question


def path(dataset_id: str) -> Path:
    return store.datasets_dir() / dataset_id / "dataset.csv"


def exists(dataset_id: str) -> bool:
    return path(dataset_id).exists()


def ids() -> list[str]:
    return [i for i in store.list_ids(store.datasets_dir()) if exists(i)]


def load(dataset_id: str) -> list[Question]:
    return csvio.read(path(dataset_id))


def sha(dataset_id: str) -> str:
    return store.sha256(path(dataset_id))


def expected_sha(dataset_id: str) -> str:
    """The hash recorded when the generator froze the dataset; a hand-supplied CSV has none, so its own hash stands."""
    return store.read_json(store.datasets_dir() / dataset_id / "manifest.json").get("csv_sha256") or sha(dataset_id)
