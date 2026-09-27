# Sink inheritance: revised E6A

The new implementation is specified in [the E6A v2 design pack](design/e6a_v2/README.md).

Coding agents start with [AGENTS.md](AGENTS.md) and [NEXT_STEPS.md](NEXT_STEPS.md). The current delivery contains design files, not a completed experiment implementation. Production runs require approval and measured feasibility gates.

`Upstream/` is the archived reference implementation. New code may borrow reviewed components with provenance, but must work when that directory is absent. Do not edit or delete it during this design handoff.

See the source audit and decision register for what is inherited, amended, proposed, or still unmeasured.

Stage 00 provides the installable `sinklab` validation package. On Windows with
Python 3.12, install the exact direct and transitive dependencies from `uv.lock`
using `uv sync --locked`. The lock selects the CUDA 12.8 Torch wheel. If the
default uv cache drive lacks space, set `UV_CACHE_DIR` to a drive with several
gigabytes free before syncing. The installed environment is local and ignored
by Git.

Run `sinklab validate --config configs/s1_c2_draft.json --seed 0` to check one
draft request. This command only validates a run selection; it does not prepare
data or train. Production validation additionally requires an approved lock
and an exact protocol digest, as described in `protocols/README.md`.
