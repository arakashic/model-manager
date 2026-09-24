from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManagedModel:
    id: str
    label: str
    container: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Warmup:
    url: str
    body: dict[str, Any]
    expect_text: str = ""
    timeout_seconds: int = 180


@dataclass(frozen=True)
class ModelPool:
    id: str
    label: str
    readiness_url: str
    models_url: str
    load_url: str | None
    models: tuple[ManagedModel, ...]
    warmup: Warmup | None = None
    poll_interval_seconds: float = 2.0
    drain_timeout_seconds: int = 300
    stop_timeout_seconds: int = 30
    start_timeout_seconds: int = 1200


@dataclass(frozen=True)
class ManagerConfig:
    pools: tuple[ModelPool, ...]


def _required_string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _positive_number(
    data: dict[str, Any], key: str, default: int | float, context: str
) -> int | float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{context}.{key} must be positive")
    return value


def load_config(path: str | Path) -> ManagerConfig:
    raw = json.loads(Path(path).read_text())
    if raw.get("version") != 1:
        raise ValueError("config.version must be 1")

    raw_pools = raw.get("pools")
    if not isinstance(raw_pools, list) or not raw_pools:
        raise ValueError("config.pools must be a non-empty array")

    pool_ids: set[str] = set()
    model_ids: set[str] = set()
    containers: set[str] = set()
    pools: list[ModelPool] = []

    for pool_index, raw_pool in enumerate(raw_pools):
        context = f"config.pools[{pool_index}]"
        if not isinstance(raw_pool, dict):
            raise ValueError(f"{context} must be an object")

        pool_id = _required_string(raw_pool, "id", context)
        if pool_id in pool_ids:
            raise ValueError(f"duplicate pool id: {pool_id}")
        pool_ids.add(pool_id)

        raw_models = raw_pool.get("models")
        if not isinstance(raw_models, list) or not raw_models:
            raise ValueError(f"{context}.models must be a non-empty array")

        models: list[ManagedModel] = []
        for model_index, raw_model in enumerate(raw_models):
            model_context = f"{context}.models[{model_index}]"
            if not isinstance(raw_model, dict):
                raise ValueError(f"{model_context} must be an object")
            model_id = _required_string(raw_model, "id", model_context)
            container = _required_string(raw_model, "container", model_context)
            if model_id in model_ids:
                raise ValueError(f"duplicate model id: {model_id}")
            if container in containers:
                raise ValueError(f"duplicate container: {container}")
            model_ids.add(model_id)
            containers.add(container)
            metadata = raw_model.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"{model_context}.metadata must be an object")
            models.append(
                ManagedModel(
                    id=model_id,
                    label=_required_string(raw_model, "label", model_context),
                    container=container,
                    metadata=metadata,
                )
            )

        warmup = None
        raw_warmup = raw_pool.get("warmup")
        if raw_warmup is not None:
            if not isinstance(raw_warmup, dict):
                raise ValueError(f"{context}.warmup must be an object")
            body = raw_warmup.get("body")
            if not isinstance(body, dict):
                raise ValueError(f"{context}.warmup.body must be an object")
            warmup = Warmup(
                url=_required_string(raw_warmup, "url", f"{context}.warmup"),
                body=body,
                expect_text=str(raw_warmup.get("expect_text", "")),
                timeout_seconds=int(
                    _positive_number(
                        raw_warmup,
                        "timeout_seconds",
                        180,
                        f"{context}.warmup",
                    )
                ),
            )

        load_url = raw_pool.get("load_url")
        if load_url is not None and (not isinstance(load_url, str) or not load_url.strip()):
            raise ValueError(f"{context}.load_url must be a non-empty string")

        pools.append(
            ModelPool(
                id=pool_id,
                label=_required_string(raw_pool, "label", context),
                readiness_url=_required_string(raw_pool, "readiness_url", context),
                models_url=_required_string(raw_pool, "models_url", context),
                load_url=load_url.strip() if isinstance(load_url, str) else None,
                models=tuple(models),
                warmup=warmup,
                poll_interval_seconds=float(
                    _positive_number(raw_pool, "poll_interval_seconds", 2.0, context)
                ),
                drain_timeout_seconds=int(
                    _positive_number(raw_pool, "drain_timeout_seconds", 300, context)
                ),
                stop_timeout_seconds=int(
                    _positive_number(raw_pool, "stop_timeout_seconds", 30, context)
                ),
                start_timeout_seconds=int(
                    _positive_number(raw_pool, "start_timeout_seconds", 1200, context)
                ),
            )
        )

    return ManagerConfig(pools=tuple(pools))
