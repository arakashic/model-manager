from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .models import ManagedModel, ManagerConfig, ModelPool
from .runtime import LifecycleRuntime


TERMINAL_STATES = {"ready", "failed", "cancelled"}


@dataclass
class Operation:
    id: str
    pool_id: str
    model_id: str
    state: str = "queued"
    phase: str = "queued"
    message: str = "Activation queued"
    progress: int = 0
    error: str | None = None
    previous_model_id: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_requested: bool = False

    def update(self, phase: str, message: str, progress: int) -> None:
        self.state = "loading"
        self.phase = phase
        self.message = message
        self.progress = progress
        self.updated_at = time.time()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pool_id": self.pool_id,
            "model_id": self.model_id,
            "state": self.state,
            "phase": self.phase,
            "message": self.message,
            "progress": self.progress,
            "error": self.error,
            "previous_model_id": self.previous_model_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "cancel_requested": self.cancel_requested,
        }


class ModelManager:
    def __init__(self, config: ManagerConfig, runtime: LifecycleRuntime) -> None:
        self.config = config
        self.runtime = runtime
        self._operations: dict[str, Operation] = {}
        self._pool_operations: dict[str, str] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._pool_by_id = {pool.id: pool for pool in config.pools}
        self._model_index = {
            model.id: (pool, model)
            for pool in config.pools
            for model in pool.models
        }

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.runtime.close()

    def operation(self, operation_id: str) -> Operation | None:
        return self._operations.get(operation_id)

    async def snapshot(self) -> dict[str, Any]:
        pools = await asyncio.gather(*(self._pool_snapshot(pool) for pool in self.config.pools))
        return {"pools": pools}

    async def _pool_snapshot(self, pool: ModelPool) -> dict[str, Any]:
        states = await asyncio.gather(
            *(self.runtime.inspect(model.container) for model in pool.models),
            return_exceptions=True,
        )
        active_operation = self._active_operation(pool.id)
        latest_operation = self._latest_operation(pool.id)
        running = [
            model
            for model, state in zip(pool.models, states)
            if not isinstance(state, BaseException) and state.running
        ]
        ready_model_id = None
        if len(running) == 1 and await self.runtime.is_ready(pool, running[0]):
            ready_model_id = running[0].id

        models = []
        for model, container_state in zip(pool.models, states):
            operation = None
            if active_operation is not None and active_operation.model_id == model.id:
                operation = active_operation
            elif (
                latest_operation is not None
                and latest_operation.model_id == model.id
                and latest_operation.state in {"failed", "cancelled"}
            ):
                operation = latest_operation
            if isinstance(container_state, BaseException):
                state = "unknown"
                detail = str(container_state)
            elif operation is not None:
                state = operation.state
                detail = operation.message
            elif model.id == ready_model_id:
                state = "ready"
                detail = "Ready"
            elif container_state.running:
                state = "starting"
                detail = "Container is running but the model is not ready"
            else:
                state = "stopped"
                detail = "Stopped"
            models.append(
                {
                    "id": model.id,
                    "label": model.label,
                    "pool_id": pool.id,
                    "container": model.container,
                    "state": state,
                    "detail": detail,
                    "active": model.id == ready_model_id,
                    "metadata": model.metadata,
                    "operation": operation.as_dict() if operation is not None else None,
                }
            )

        return {
            "id": pool.id,
            "label": pool.label,
            "active_model_id": ready_model_id,
            "models": models,
            "operation": latest_operation.as_dict() if latest_operation is not None else None,
        }

    def _latest_operation(self, pool_id: str) -> Operation | None:
        operation_id = self._pool_operations.get(pool_id)
        if operation_id is None:
            return None
        return self._operations.get(operation_id)

    def _active_operation(self, pool_id: str) -> Operation | None:
        operation_id = self._pool_operations.get(pool_id)
        if operation_id is None:
            return None
        operation = self._operations.get(operation_id)
        if operation is None or operation.state in TERMINAL_STATES:
            return None
        return operation

    async def activate(self, model_id: str) -> Operation:
        target = self._model_index.get(model_id)
        if target is None:
            raise KeyError(model_id)
        pool, model = target
        active_operation = self._active_operation(pool.id)
        if active_operation is not None:
            if active_operation.model_id == model_id:
                return active_operation
            raise RuntimeError(f"pool {pool.id} already has an activation in progress")

        state = await self.runtime.inspect(model.container)
        if state.running and await self.runtime.is_ready(pool, model):
            operation = Operation(
                id=str(uuid.uuid4()),
                pool_id=pool.id,
                model_id=model.id,
                state="ready",
                phase="ready",
                message="Model is already ready",
                progress=100,
            )
            self._store_operation(operation)
            self._pool_operations[pool.id] = operation.id
            return operation

        operation = Operation(id=str(uuid.uuid4()), pool_id=pool.id, model_id=model.id)
        self._store_operation(operation)
        self._pool_operations[pool.id] = operation.id
        task = asyncio.create_task(self._run_activation(pool, model, operation))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return operation

    def _store_operation(self, operation: Operation) -> None:
        self._operations[operation.id] = operation
        if len(self._operations) <= 100:
            return
        oldest = min(self._operations.values(), key=lambda item: item.updated_at)
        if oldest.state in TERMINAL_STATES:
            self._operations.pop(oldest.id, None)

    def cancel(self, operation_id: str) -> Operation:
        operation = self._operations.get(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        if operation.state not in TERMINAL_STATES:
            operation.cancel_requested = True
            operation.message = "Cancellation requested"
            operation.updated_at = time.time()
        return operation

    async def _run_activation(
        self, pool: ModelPool, target: ManagedModel, operation: Operation
    ) -> None:
        previous = None
        target_started = False
        try:
            operation.update("inspect", "Inspecting the model pool", 5)
            running = await self._running_models(pool)
            if len(running) > 1:
                raise RuntimeError("multiple managed model containers are running")
            previous = running[0] if running else None
            operation.previous_model_id = previous.id if previous is not None else None

            if previous is not None and previous.id != target.id:
                await self._drain(pool, operation)
                self._check_cancel(operation)
                operation.update("stopping", f"Stopping {previous.label}", 20)
                await self.runtime.set_restart_policy(previous.container, "no")
                await self.runtime.stop(previous.container, pool.stop_timeout_seconds)

            self._check_cancel(operation)
            operation.update("starting", f"Starting {target.label}", 35)
            await self.runtime.set_restart_policy(target.container, "unless-stopped")
            await self.runtime.start(target.container)
            target_started = True

            await self._wait_ready(pool, target, operation)
            self._check_cancel(operation)
            operation.update("warming", f"Warming up {target.label}", 92)
            await self.runtime.warmup(pool, target)
            self._check_cancel(operation)
            operation.state = "ready"
            operation.phase = "ready"
            operation.message = f"{target.label} is ready"
            operation.progress = 100
            operation.updated_at = time.time()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            rollback_error = await self._rollback(pool, target, previous, target_started)
            operation.state = "cancelled" if operation.cancel_requested else "failed"
            operation.phase = operation.state
            operation.error = str(exc)
            operation.message = (
                "Activation cancelled" if operation.cancel_requested else "Activation failed"
            )
            if rollback_error is not None:
                operation.error = f"{operation.error}; rollback failed: {rollback_error}"
            operation.updated_at = time.time()

    async def _running_models(self, pool: ModelPool) -> list[ManagedModel]:
        states = await asyncio.gather(
            *(self.runtime.inspect(model.container) for model in pool.models)
        )
        return [model for model, state in zip(pool.models, states) if state.running]

    async def _drain(self, pool: ModelPool, operation: Operation) -> None:
        deadline = asyncio.get_running_loop().time() + pool.drain_timeout_seconds
        while True:
            self._check_cancel(operation)
            active = await self.runtime.active_requests(pool)
            if active == 0:
                return
            if active is None:
                raise RuntimeError("could not determine active request count")
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(f"timed out waiting for {active} active requests")
            operation.update("draining", f"Waiting for {active} active requests", 12)
            await asyncio.sleep(pool.poll_interval_seconds)

    async def _wait_ready(
        self, pool: ModelPool, target: ManagedModel, operation: Operation
    ) -> None:
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + pool.start_timeout_seconds
        while loop.time() < deadline:
            self._check_cancel(operation)
            state = await self.runtime.inspect(target.container)
            if not state.running:
                raise RuntimeError(f"container exited with status {state.status}")
            if await self.runtime.is_ready(pool, target):
                return
            elapsed = loop.time() - started
            progress = min(88, 40 + int(48 * elapsed / pool.start_timeout_seconds))
            operation.update("loading", f"Loading {target.label}", progress)
            await asyncio.sleep(pool.poll_interval_seconds)
        raise RuntimeError("timed out waiting for model readiness")

    @staticmethod
    def _check_cancel(operation: Operation) -> None:
        if operation.cancel_requested:
            raise RuntimeError("activation cancelled")

    async def _rollback(
        self,
        pool: ModelPool,
        target: ManagedModel,
        previous: ManagedModel | None,
        target_started: bool,
    ) -> str | None:
        try:
            if target_started:
                await self.runtime.stop(target.container, pool.stop_timeout_seconds)
                await self.runtime.set_restart_policy(target.container, "no")
            if previous is None or previous.id == target.id:
                return None
            await self.runtime.set_restart_policy(previous.container, "unless-stopped")
            await self.runtime.start(previous.container)
            deadline = asyncio.get_running_loop().time() + pool.start_timeout_seconds
            while asyncio.get_running_loop().time() < deadline:
                if await self.runtime.is_ready(pool, previous):
                    return None
                state = await self.runtime.inspect(previous.container)
                if not state.running:
                    raise RuntimeError(f"rollback container exited with status {state.status}")
                await asyncio.sleep(pool.poll_interval_seconds)
            raise RuntimeError("rollback model did not become ready")
        except Exception as exc:
            return str(exc)
