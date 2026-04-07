from __future__ import annotations


DATASET_NAME_ALIASES = {
    # The legacy built-in `daily_dialog` loader relies on a dataset script.
    # Recent `datasets` versions disable dataset scripts by default, so point
    # to a Hub-hosted mirror instead.
    "daily_dialog": "Akhil391/daily_dialog",
}


def resolve_dataset_name(dataset_name: str | None) -> str | None:
    if dataset_name is None:
        return None
    return DATASET_NAME_ALIASES.get(dataset_name, dataset_name)
