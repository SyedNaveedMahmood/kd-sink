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

Stage 01 preparation is explicit and offline. Supply an operator-owned artifact
directory outside this repository and already-local, revision-pinned source
files; none of these commands downloads a corpus or starts training. A corpus
JSONL row has exactly `source_index` (integer), `document_id` (string), and
`text` (string). The tokenizer directory must contain saved local tokenizer
files and an EOS token. Record the actual source revision and license after
verifying them; the CLI does not infer either.

```powershell
sinklab prepare-corpus --input-jsonl <local-owt.jsonl> --tokenizer-dir <local-gpt2-tokenizer> --dataset-id openwebtext --dataset-revision <pinned-revision> --license-id <verified-license> --tokenizer-id gpt2 --tokenizer-revision <pinned-revision> --out-dir <external-artifact-dir>
sinklab prepare-init --config <verified-gpt2-medium-config.json> --study S1 --seed 0 --out-dir <external-artifact-dir>
sinklab prepare-order --corpus <corpus-hash.json> --seed 0 --updates 0 --out-dir <external-artifact-dir>
sinklab prepare-panels --corpus <corpus-hash.json> --out-dir <external-artifact-dir>
sinklab prepare-domains --tokenizer-dir <local-gpt2-tokenizer> --sst2-jsonl <local-sst2.jsonl> --sst2-revision <pinned-revision> --gsm8k-jsonl <local-gsm8k.jsonl> --gsm8k-revision <pinned-revision> --humaneval-jsonl <local-humaneval.jsonl> --humaneval-revision <pinned-revision> --out-dir <external-artifact-dir>
```

The domain JSONL rows must carry `split` and respectively `sentence` (SST-2
validation), `question` (GSM8K test), or `prompt` (HumanEval test). Other row
fields are never copied into panel inputs. `prepare-init --fixture` permits a
tiny local test config; real S1/S3 commands enforce the declared student
dimensions. Prepared manifest filenames contain their payload SHA-256. Use
`sinklab.data.load_corpus` with the expected tokenizer file hash and manifest
digest before consuming blocks. Production revisions, licenses, approvals,
and hardware readiness are not established by these synthetic Stage 01 tests.
