import json
import random

import pytest

from sinklab.order import (INPUT_TOKENS_PER_UPDATE, SHIFTED_TARGETS_PER_UPDATE,
                           OrderError, UpdateOrder, microbatches)


def test_updates_microbatches_epochs_resume_and_extension():
    ids = [f"block-{i:03}" for i in range(100)]
    before = random.getstate()
    order = UpdateOrder(ids, seed=17)
    assert random.getstate() == before
    first = order.take_update()
    assert len(first) == 64
    assert [x["block_id"] for batch in microbatches(first, 8) for x in batch] == [x["block_id"] for x in first]
    assert [x["block_id"] for batch in microbatches(first, 16) for x in batch] == [x["block_id"] for x in first]
    assert INPUT_TOKENS_PER_UPDATE == 64 * 128 == 8192
    assert SHIFTED_TARGETS_PER_UPDATE == 64 * 127 == 8128
    snapshot = json.loads(json.dumps(order.snapshot()))
    resumed = UpdateOrder.resume(ids, snapshot)
    rest = [order.take_update() for _ in range(4)]
    assert rest == [resumed.take_update() for _ in range(4)]
    assert order.snapshot()["payload"]["prefix_sha256"] == resumed.snapshot()["payload"]["prefix_sha256"]
    assert rest[0][36]["epoch"] == 1
    uninterrupted = UpdateOrder(ids, seed=17)
    assert [uninterrupted.take_update() for _ in range(5)] == [first] + rest
    assert [x for update in rest for x in update] == [x for update in [UpdateOrder.resume(ids, snapshot).take_update()] for x in update] + [x for update in rest[1:] for x in update]
    assert random.getstate() == before


def test_order_rejects_mismatch_and_invalid_microbatch():
    order = UpdateOrder([str(i) for i in range(65)], seed=1)
    snapshot = order.snapshot()
    with pytest.raises(OrderError, match="manifest"):
        UpdateOrder.resume([str(i) for i in range(64)], snapshot)
    with pytest.raises(OrderError, match="microbatch"):
        microbatches(order.take_update(), 7)
