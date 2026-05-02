from __future__ import annotations

from typing import Protocol, runtime_checkable

from autopilot.models import ModelConfig, Response


@runtime_checkable
class Provider(Protocol):
    name: str

    async def complete(self, prompt: str, config: ModelConfig) -> Response: ...
