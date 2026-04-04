from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FunscriptAction:
    at: int
    pos: int


@dataclass(slots=True)
class FunscriptMetadata:
    creator: str = "bREadbeats PMV Generator"
    title: str = ""
    duration: int = 0
    description: str = ""
    performers: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    type: str = "basic"
    # Stored as the top-level JSON key "pmv_params".
    parameters: dict[str, Any] = field(default_factory=dict)


def actions_to_dict_list(actions: list[FunscriptAction]) -> list[dict[str, int]]:
    """Convert actions to Funscript JSON action dictionaries."""
    return [{"at": int(a.at), "pos": int(a.pos)} for a in actions]


def dict_list_to_actions(data: list[dict[str, Any]]) -> list[FunscriptAction]:
    """Convert action dictionaries into dataclass actions."""
    actions: list[FunscriptAction] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        at = int(item.get("at", 0))
        pos = int(item.get("pos", 0))
        actions.append(FunscriptAction(at=at, pos=pos))
    return actions


def write_funscript(
    path: str | Path,
    actions: list[FunscriptAction],
    metadata: FunscriptMetadata | None = None,
    inverted: bool = False,
    range_: int = 100,
) -> None:
    """Write a Funscript 1.0 JSON file."""
    target = Path(path)
    meta = metadata or FunscriptMetadata()

    payload: dict[str, Any] = {
        "version": "1.0",
        "inverted": bool(inverted),
        "range": int(range_),
        "actions": actions_to_dict_list(actions),
        "metadata": {
            "creator": str(meta.creator),
            "title": str(meta.title),
            "duration": int(meta.duration),
            "description": str(meta.description),
            "performers": [str(v) for v in meta.performers],
            "tags": [str(v) for v in meta.tags],
            "type": str(meta.type),
        },
    }
    if meta.parameters:
        payload["pmv_params"] = dict(meta.parameters)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_funscript(path: str | Path) -> tuple[list[FunscriptAction], FunscriptMetadata]:
    """Read a Funscript JSON file and return actions plus metadata."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Invalid funscript payload: expected top-level object")

    actions_raw = raw.get("actions", [])
    if not isinstance(actions_raw, list):
        raise ValueError("Invalid funscript payload: 'actions' must be a list")

    metadata_raw = raw.get("metadata", {})
    if not isinstance(metadata_raw, dict):
        metadata_raw = {}

    params_raw = raw.get("pmv_params", {})
    if not isinstance(params_raw, dict):
        params_raw = {}

    metadata = FunscriptMetadata(
        creator=str(metadata_raw.get("creator", "bREadbeats PMV Generator")),
        title=str(metadata_raw.get("title", "")),
        duration=int(metadata_raw.get("duration", 0) or 0),
        description=str(metadata_raw.get("description", "")),
        performers=[str(v) for v in metadata_raw.get("performers", []) if v is not None],
        tags=[str(v) for v in metadata_raw.get("tags", []) if v is not None],
        type=str(metadata_raw.get("type", "basic")),
        parameters=dict(params_raw),
    )

    return dict_list_to_actions(actions_raw), metadata


def write_csv(path: str | Path, actions: list[FunscriptAction]) -> None:
    """Write CSV as columns: at_ms, position."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["at_ms", "position"])
        for action in actions:
            writer.writerow([int(action.at), int(action.pos)])


def read_csv(path: str | Path) -> list[FunscriptAction]:
    """Read CSV action data from either headered or raw 2-column files."""
    source = Path(path)

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if row]

    if not rows:
        return []

    first = [c.strip().lower() for c in rows[0]]
    has_header = len(first) >= 2 and (
        first[0] in {"at", "at_ms", "time", "time_ms"}
        or first[1] in {"pos", "position"}
    )

    data_rows = rows[1:] if has_header else rows
    actions: list[FunscriptAction] = []
    for row in data_rows:
        if len(row) < 2:
            continue
        try:
            at = int(float(row[0].strip()))
            pos = int(float(row[1].strip()))
        except (TypeError, ValueError):
            continue
        actions.append(FunscriptAction(at=at, pos=pos))
    return actions
