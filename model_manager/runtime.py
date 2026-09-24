from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

from aiohttp import ClientResponse, ClientSession, ClientTimeout, UnixConnector

from .models import ManagedModel, ModelPool


@dataclass(frozen=True)
class ContainerState:
    status: str
    running: bool


class LifecycleRuntime(Protocol):
    async def inspect(self, container: str) -> ContainerState: ...

    async def set_restart_policy(self, container: str, policy: str) -> None: ...

    async def start(self, container: str) -> None: ...

    async def stop(self, container: str, timeout_seconds: int) -> None: ...

    async def is_ready(self, pool: ModelPool, model: ManagedModel) -> bool: ...

    async def active_requests(self, pool: ModelPool) -> int | None: ...

    async def warmup(self, pool: ModelPool, model: ManagedModel) -> None: ...

    async def close(self) -> None: ...


def _replace_model_id(value: Any, model_id: str) -> Any:
    if isinstance(value, str):
        return value.replace("${MODEL_ID}", model_id)
    if isinstance(value, list):
        return [_replace_model_id(item, model_id) for item in value]
    if isinstance(value, dict):
        return {key: _replace_model_id(item, model_id) for key, item in value.items()}
    return value


class DockerRuntime:
    def __init__(self, socket_path: str = "/var/run/docker.sock") -> None:
        self._socket_path = socket_path
        self._docker: ClientSession | None = None
        self._http: ClientSession | None = None

    def _sessions(self) -> tuple[ClientSession, ClientSession]:
        if self._docker is None:
            self._docker = ClientSession(connector=UnixConnector(path=self._socket_path))
        if self._http is None:
            self._http = ClientSession(timeout=ClientTimeout(total=10))
        return self._docker, self._http

    async def _docker_request(
        self,
        method: str,
        path: str,
        *,
        expected: tuple[int, ...],
        body: dict[str, Any] | None = None,
    ) -> ClientResponse:
        docker, _http = self._sessions()
        response = await docker.request(
            method,
            f"http://docker{path}",
            json=body,
        )
        if response.status not in expected:
            detail = (await response.text())[:500]
            response.release()
            raise RuntimeError(f"Docker {method} {path} returned {response.status}: {detail}")
        return response

    async def inspect(self, container: str) -> ContainerState:
        name = quote(container, safe="")
        response = await self._docker_request(
            "GET", f"/containers/{name}/json", expected=(200,)
        )
        data = await response.json()
        return ContainerState(
            status=str(data.get("State", {}).get("Status", "unknown")),
            running=bool(data.get("State", {}).get("Running", False)),
        )

    async def set_restart_policy(self, container: str, policy: str) -> None:
        name = quote(container, safe="")
        response = await self._docker_request(
            "POST",
            f"/containers/{name}/update",
            expected=(200,),
            body={"RestartPolicy": {"Name": policy, "MaximumRetryCount": 0}},
        )
        await response.read()

    async def start(self, container: str) -> None:
        name = quote(container, safe="")
        response = await self._docker_request(
            "POST", f"/containers/{name}/start", expected=(204, 304)
        )
        await response.read()

    async def stop(self, container: str, timeout_seconds: int) -> None:
        name = quote(container, safe="")
        response = await self._docker_request(
            "POST",
            f"/containers/{name}/stop?t={timeout_seconds}",
            expected=(204, 304),
        )
        await response.read()

    async def is_ready(self, pool: ModelPool, model: ManagedModel) -> bool:
        _docker, http = self._sessions()
        try:
            async with http.get(pool.readiness_url) as health:
                if health.status < 200 or health.status >= 300:
                    return False
                await health.read()
            async with http.get(pool.models_url) as models_response:
                if models_response.status < 200 or models_response.status >= 300:
                    return False
                payload = await models_response.json()
        except (OSError, asyncio.TimeoutError, json.JSONDecodeError):
            return False

        models = payload.get("data")
        if not isinstance(models, list):
            return False
        return any(item.get("id") == model.id for item in models if isinstance(item, dict))

    async def active_requests(self, pool: ModelPool) -> int | None:
        if pool.load_url is None:
            return 0
        _docker, http = self._sessions()
        try:
            async with http.get(pool.load_url) as response:
                if response.status < 200 or response.status >= 300:
                    return None
                payload = await response.json()
        except (OSError, asyncio.TimeoutError, json.JSONDecodeError):
            return None

        if not isinstance(payload, list):
            return None
        total = 0
        for item in payload:
            if not isinstance(item, dict):
                continue
            total += int(item.get("num_reqs", 0))
            total += int(item.get("num_waiting_reqs", 0))
        return total

    async def warmup(self, pool: ModelPool, model: ManagedModel) -> None:
        if pool.warmup is None:
            return
        body = _replace_model_id(pool.warmup.body, model.id)
        timeout = ClientTimeout(total=pool.warmup.timeout_seconds)
        _docker, http = self._sessions()
        async with http.post(pool.warmup.url, json=body, timeout=timeout) as response:
            text = await response.text()
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"warm-up returned {response.status}: {text[:500]}")
            if pool.warmup.expect_text and pool.warmup.expect_text.lower() not in text.lower():
                raise RuntimeError("warm-up response did not contain the expected text")

    async def close(self) -> None:
        if self._docker is not None:
            await self._docker.close()
        if self._http is not None:
            await self._http.close()
