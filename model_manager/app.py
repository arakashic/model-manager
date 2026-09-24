from __future__ import annotations

import hmac
import os
from typing import Awaitable, Callable

from aiohttp import web

from .models import load_config
from .runtime import DockerRuntime
from .service import ModelManager


Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


def json_error(message: str, status: int) -> web.Response:
    return web.json_response({"error": {"message": message}}, status=status)


def create_app(manager: ModelManager, api_key: str) -> web.Application:
    @web.middleware
    async def authenticate(request: web.Request, handler: Handler) -> web.StreamResponse:
        if not request.path.startswith("/api/"):
            return await handler(request)
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {api_key}"):
            return json_error("Unauthorized", 401)
        return await handler(request)

    app = web.Application(middlewares=[authenticate])
    app["manager"] = manager

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def models(_request: web.Request) -> web.Response:
        return web.json_response(await manager.snapshot())

    async def activate(request: web.Request) -> web.Response:
        try:
            operation = await manager.activate(request.match_info["model_id"])
        except KeyError:
            return json_error("Unknown model", 404)
        except RuntimeError as exc:
            return json_error(str(exc), 409)
        status = 200 if operation.state == "ready" else 202
        return web.json_response({"operation": operation.as_dict()}, status=status)

    async def operation(request: web.Request) -> web.Response:
        item = manager.operation(request.match_info["operation_id"])
        if item is None:
            return json_error("Unknown operation", 404)
        return web.json_response({"operation": item.as_dict()})

    async def cancel(request: web.Request) -> web.Response:
        try:
            item = manager.cancel(request.match_info["operation_id"])
        except KeyError:
            return json_error("Unknown operation", 404)
        return web.json_response({"operation": item.as_dict()})

    async def cleanup(_app: web.Application) -> None:
        await manager.close()

    app.router.add_get("/health", health)
    app.router.add_get("/api/v1/models", models)
    app.router.add_post("/api/v1/models/{model_id}/activate", activate)
    app.router.add_get("/api/v1/operations/{operation_id}", operation)
    app.router.add_post("/api/v1/operations/{operation_id}/cancel", cancel)
    app.on_cleanup.append(cleanup)
    return app


def main() -> None:
    api_key = os.environ.get("MODEL_MANAGER_API_KEY", "")
    if not api_key:
        raise RuntimeError("MODEL_MANAGER_API_KEY is required")
    config_path = os.environ.get("MODEL_MANAGER_CONFIG", "/config/config.json")
    socket_path = os.environ.get("MODEL_MANAGER_DOCKER_SOCKET", "/var/run/docker.sock")
    manager = ModelManager(load_config(config_path), DockerRuntime(socket_path))
    web.run_app(
        create_app(manager, api_key),
        host=os.environ.get("MODEL_MANAGER_HOST", "127.0.0.1"),
        port=int(os.environ.get("MODEL_MANAGER_PORT", "8002")),
    )


if __name__ == "__main__":
    main()
