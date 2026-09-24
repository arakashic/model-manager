# Model Manager

`model-manager` provides authenticated lifecycle control for allowlisted Docker
containers that expose OpenAI-compatible inference APIs. It is model-agnostic:
model IDs, containers, readiness checks, request-load checks, warm-up requests,
and pool membership come from JSON configuration.

Only one model may run in each pool. Switching waits for active requests,
stops the previous container, starts and health-checks the selected container,
runs an optional warm-up, and rolls back if activation fails or is cancelled.
The service is not specific to Qwen, SGLang, or a particular GPU topology.

The service requires access to the Docker socket and must not be exposed without
its bearer token. The API cannot execute arbitrary container names or commands;
it operates only on entries in its read-only configuration.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Unauthenticated process health |
| `GET` | `/api/v1/models` | Pools, model states, and active operations |
| `POST` | `/api/v1/models/{id}/activate` | Activate an allowlisted model |
| `GET` | `/api/v1/operations/{id}` | Activation progress |
| `POST` | `/api/v1/operations/{id}/cancel` | Cancel and roll back an activation |

All `/api/` routes require `Authorization: Bearer $MODEL_MANAGER_API_KEY`.
