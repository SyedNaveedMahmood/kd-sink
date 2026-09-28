"""Horizon-independent, resumable effective-update block stream."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from .provenance import canonical_json_bytes, seal_payload, verify_envelope

SEQUENCES_PER_UPDATE = 64
TOKENS_PER_SEQUENCE = 128
INPUT_TOKENS_PER_UPDATE = 8192
SHIFTED_TARGETS_PER_UPDATE = 8128


class OrderError(ValueError):
    pass


def _tuple_tree(value: Any) -> Any:
    return tuple(_tuple_tree(x) for x in value) if isinstance(value, list) else value


class UpdateOrder:
    def __init__(self, block_ids: list[str], *, seed: int,
                 scheme: str = "v2-horizon-independent"):
        if type(seed) is not int or seed < 0 or not block_ids or len(set(block_ids)) != len(block_ids):
            raise OrderError("explicit seed and unique nonempty training block IDs required")
        if scheme not in {"v2-horizon-independent", "upstream-owt-epoch-v1"}:
            raise OrderError("unsupported update-order scheme")
        self.block_ids = tuple(block_ids)
        self.block_set_hash = hashlib.sha256(canonical_json_bytes({"block_ids": block_ids})).hexdigest()
        self.seed = seed
        self.scheme = scheme
        self.rng = random.Random(seed)
        self.epoch = 0
        self.cursor = 0
        self.permutation = list(block_ids)
        self._shuffle_epoch()
        self.presentations = 0
        self.prefix_sha256 = hashlib.sha256(b"e6a-v2-update-order-v1").hexdigest()

    def _next(self) -> dict[str, Any]:
        if self.cursor == len(self.permutation):
            self.epoch += 1
            self.cursor = 0
            self.permutation = list(self.block_ids)
            self._shuffle_epoch()
        item = {"epoch": self.epoch, "epoch_cursor": self.cursor,
                "block_id": self.permutation[self.cursor]}
        self.cursor += 1
        self.presentations += 1
        self.prefix_sha256 = hashlib.sha256(canonical_json_bytes({
            "previous": self.prefix_sha256, "presentation": self.presentations, "item": item})).hexdigest()
        return item

    def _shuffle_epoch(self) -> None:
        if self.scheme == "upstream-owt-epoch-v1":
            random.Random((self.seed + 1) * 100003 + self.epoch).shuffle(self.permutation)
        else:
            self.rng.shuffle(self.permutation)

    def take_update(self) -> list[dict[str, Any]]:
        return [self._next() for _ in range(SEQUENCES_PER_UPDATE)]

    def snapshot(self) -> dict[str, Any]:
        return seal_payload({"kind": "update-order-v1", "scheme": self.scheme,
            "block_set_sha256": self.block_set_hash,
            "seed": self.seed, "epoch": self.epoch, "cursor": self.cursor,
            "permutation": self.permutation, "rng_state": self.rng.getstate(),
            "presentations": self.presentations, "prefix_sha256": self.prefix_sha256})

    @classmethod
    def resume(cls, block_ids: list[str], snapshot: dict[str, Any], *,
               expected_scheme: str | None = None) -> "UpdateOrder":
        payload, _ = verify_envelope(snapshot)
        if payload.get("kind") != "update-order-v1":
            raise OrderError("unsupported update-order state")
        scheme = payload.get("scheme", "v2-horizon-independent")
        if expected_scheme is not None and scheme != expected_scheme:
            raise OrderError("update-order scheme changed on resume")
        state = cls(block_ids, seed=payload["seed"], scheme=scheme)
        if state.block_set_hash != payload["block_set_sha256"]:
            raise OrderError("training block manifest mismatch")
        if (type(payload["epoch"]) is not int or payload["epoch"] < 0 or
                type(payload["cursor"]) is not int or not 0 <= payload["cursor"] <= len(block_ids) or
                sorted(payload["permutation"]) != sorted(block_ids) or
                type(payload["presentations"]) is not int or
                payload["presentations"] != payload["epoch"] * len(block_ids) + payload["cursor"]):
            raise OrderError("invalid update-order cursor or permutation")
        state.epoch = payload["epoch"]
        state.cursor = payload["cursor"]
        state.permutation = payload["permutation"]
        state.presentations = payload["presentations"]
        state.prefix_sha256 = payload["prefix_sha256"]
        try:
            state.rng.setstate(_tuple_tree(payload["rng_state"]))
        except (TypeError, ValueError) as exc:
            raise OrderError("invalid dedicated RNG state") from exc
        return state


def microbatches(update: list[dict[str, Any]], microbatch_size: int) -> list[list[dict[str, Any]]]:
    if (len(update) != SEQUENCES_PER_UPDATE or type(microbatch_size) is not int or
            microbatch_size < 1 or SEQUENCES_PER_UPDATE % microbatch_size):
        raise OrderError("microbatch must divide one 64-sequence effective update")
    return [update[i:i + microbatch_size] for i in range(0, 64, microbatch_size)]
