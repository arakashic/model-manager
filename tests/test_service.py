import asyncio
import unittest

from model_manager.models import ManagedModel, ManagerConfig, ModelPool, Warmup
from model_manager.runtime import ContainerState
from model_manager.service import ModelManager


class FakeRuntime:
    def __init__(
        self,
        states,
        ready=None,
        active_requests=0,
        fail_warmup=False,
        warmup_started=None,
        warmup_release=None,
    ):
        self.states = states
        self.ready = set(ready or [])
        self.request_count = active_requests
        self.fail_warmup = fail_warmup
        self.warmup_started = warmup_started
        self.warmup_release = warmup_release
        self.actions = []

    async def inspect(self, container):
        return self.states[container]

    async def set_restart_policy(self, container, policy):
        self.actions.append(("restart", container, policy))

    async def start(self, container):
        self.actions.append(("start", container))
        self.states[container] = ContainerState(status="running", running=True)
        self.ready.add(container)

    async def stop(self, container, timeout_seconds):
        self.actions.append(("stop", container, timeout_seconds))
        self.states[container] = ContainerState(status="exited", running=False)
        self.ready.discard(container)

    async def is_ready(self, pool, model):
        return model.container in self.ready

    async def active_requests(self, pool):
        return self.request_count

    async def warmup(self, pool, model):
        self.actions.append(("warmup", model.container))
        if self.warmup_started is not None:
            self.warmup_started.set()
        if self.warmup_release is not None:
            await self.warmup_release.wait()
        if self.fail_warmup:
            raise RuntimeError("warm-up failed")

    async def close(self):
        pass


def config():
    stock = ManagedModel(id="stock", label="Stock", container="stock-container")
    alternate = ManagedModel(id="alternate", label="Alternate", container="alt-container")
    pool = ModelPool(
        id="gpu-0",
        label="GPU 0",
        readiness_url="http://localhost/health",
        models_url="http://localhost/v1/models",
        load_url="http://localhost/get_load",
        models=(stock, alternate),
        warmup=Warmup(url="http://localhost/chat", body={"model": "${MODEL_ID}"}),
        poll_interval_seconds=0.001,
        drain_timeout_seconds=1,
        stop_timeout_seconds=1,
        start_timeout_seconds=1,
    )
    return ManagerConfig(pools=(pool,))


async def wait_terminal(manager, operation):
    for _ in range(100):
        if operation.state in {"ready", "failed", "cancelled"}:
            return
        await asyncio.sleep(0.001)
    raise AssertionError("operation did not finish")


class ModelManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_reports_ready_and_stopped_models(self):
        runtime = FakeRuntime(
            {
                "stock-container": ContainerState(status="running", running=True),
                "alt-container": ContainerState(status="created", running=False),
            },
            ready={"stock-container"},
        )
        manager = ModelManager(config(), runtime)

        snapshot = await manager.snapshot()

        self.assertEqual(snapshot["pools"][0]["active_model_id"], "stock")
        self.assertEqual(
            [item["state"] for item in snapshot["pools"][0]["models"]],
            ["ready", "stopped"],
        )

    async def test_switches_models_after_draining(self):
        runtime = FakeRuntime(
            {
                "stock-container": ContainerState(status="running", running=True),
                "alt-container": ContainerState(status="created", running=False),
            },
            ready={"stock-container"},
        )
        manager = ModelManager(config(), runtime)

        operation = await manager.activate("alternate")
        await wait_terminal(manager, operation)

        self.assertEqual(operation.state, "ready")
        self.assertEqual(
            runtime.actions,
            [
                ("restart", "stock-container", "no"),
                ("stop", "stock-container", 1),
                ("restart", "alt-container", "unless-stopped"),
                ("start", "alt-container"),
                ("warmup", "alt-container"),
            ],
        )

    async def test_rolls_back_when_warmup_fails(self):
        runtime = FakeRuntime(
            {
                "stock-container": ContainerState(status="running", running=True),
                "alt-container": ContainerState(status="created", running=False),
            },
            ready={"stock-container"},
            fail_warmup=True,
        )
        manager = ModelManager(config(), runtime)

        operation = await manager.activate("alternate")
        await wait_terminal(manager, operation)

        self.assertEqual(operation.state, "failed")
        self.assertIn("warm-up failed", operation.error)
        self.assertEqual(runtime.states["stock-container"].running, True)
        self.assertEqual(runtime.states["alt-container"].running, False)
        snapshot = await manager.snapshot()
        alternate = snapshot["pools"][0]["models"][1]
        self.assertEqual(alternate["state"], "failed")
        self.assertIn("warm-up failed", alternate["operation"]["error"])

    async def test_refuses_to_switch_with_active_requests(self):
        runtime = FakeRuntime(
            {
                "stock-container": ContainerState(status="running", running=True),
                "alt-container": ContainerState(status="created", running=False),
            },
            ready={"stock-container"},
            active_requests=1,
        )
        cfg = config()
        pool = cfg.pools[0]
        cfg = ManagerConfig(
            pools=(
                ModelPool(
                    **{
                        **pool.__dict__,
                        "drain_timeout_seconds": 0.003,
                    }
                ),
            )
        )
        manager = ModelManager(cfg, runtime)

        operation = await manager.activate("alternate")
        await wait_terminal(manager, operation)

        self.assertEqual(operation.state, "failed")
        self.assertIn("active requests", operation.error)
        self.assertNotIn(("stop", "stock-container", 1), runtime.actions)

    async def test_activation_of_ready_model_is_noop(self):
        runtime = FakeRuntime(
            {
                "stock-container": ContainerState(status="running", running=True),
                "alt-container": ContainerState(status="created", running=False),
            },
            ready={"stock-container"},
        )
        manager = ModelManager(config(), runtime)

        operation = await manager.activate("stock")

        self.assertEqual(operation.state, "ready")
        self.assertEqual(runtime.actions, [])
        snapshot = await manager.snapshot()
        self.assertEqual(snapshot["pools"][0]["operation"]["id"], operation.id)

        repeated = await manager.activate("stock")
        self.assertEqual(repeated.id, operation.id)

    async def test_concurrent_activation_reuses_operation(self):
        runtime = FakeRuntime(
            {
                "stock-container": ContainerState(status="running", running=True),
                "alt-container": ContainerState(status="created", running=False),
            },
            ready={"stock-container"},
        )
        manager = ModelManager(config(), runtime)

        first, second = await asyncio.gather(
            manager.activate("alternate"),
            manager.activate("alternate"),
        )
        await wait_terminal(manager, first)

        self.assertEqual(first.id, second.id)
        self.assertEqual(
            [action for action in runtime.actions if action == ("start", "alt-container")],
            [("start", "alt-container")],
        )

    async def test_conflicting_activation_is_rejected(self):
        runtime = FakeRuntime(
            {
                "stock-container": ContainerState(status="running", running=True),
                "alt-container": ContainerState(status="created", running=False),
            },
            ready={"stock-container"},
        )
        manager = ModelManager(config(), runtime)

        operation = await manager.activate("alternate")
        with self.assertRaisesRegex(RuntimeError, "activation in progress"):
            await manager.activate("stock")
        await wait_terminal(manager, operation)

    async def test_cancellation_during_warmup_rolls_back(self):
        warmup_started = asyncio.Event()
        warmup_release = asyncio.Event()
        runtime = FakeRuntime(
            {
                "stock-container": ContainerState(status="running", running=True),
                "alt-container": ContainerState(status="created", running=False),
            },
            ready={"stock-container"},
            warmup_started=warmup_started,
            warmup_release=warmup_release,
        )
        manager = ModelManager(config(), runtime)

        operation = await manager.activate("alternate")
        await asyncio.wait_for(warmup_started.wait(), timeout=1)
        manager.cancel(operation.id)
        warmup_release.set()
        await wait_terminal(manager, operation)

        self.assertEqual(operation.state, "cancelled")
        self.assertEqual(runtime.states["stock-container"].running, True)
        self.assertEqual(runtime.states["alt-container"].running, False)


if __name__ == "__main__":
    unittest.main()
