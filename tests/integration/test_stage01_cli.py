import json
import subprocess
import sys

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, PreTrainedTokenizerFast

from sinklab.data import document_hash, document_split, load_manifest, tokenizer_files_hash, validate_corpus
from sinklab.order import UpdateOrder


def _run(*args):
    return subprocess.run([sys.executable, "-m", "sinklab", *map(str, args)],
                          capture_output=True, text=True)


def _source_text(split):
    for n in range(10000):
        text = f"z {n} " + "z " * 150
        if document_split(document_hash(text)) == split:
            return text
    raise AssertionError("missing synthetic split")


def test_explicit_offline_preparation_commands(tmp_path):
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<eos>": 1, "z": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"source_index": i, "document_id": split,
        "text": _source_text(split)}) + "\n" for i, split in enumerate(
            ("training", "calibration", "evaluation"))), encoding="utf-8")
    out = tmp_path / "artifacts"
    corpus_run = _run("prepare-corpus", "--input-jsonl", source, "--tokenizer-dir", tokenizer_dir,
        "--dataset-id", "synthetic", "--dataset-revision", "fixture-v1",
        "--license-id", "fixture-only", "--tokenizer-id", "tiny-local",
        "--tokenizer-revision", "fixture-v1", "--out-dir", out)
    assert corpus_run.returncode == 0, corpus_run.stderr
    corpus_path = json.loads(corpus_run.stdout)["path"]
    payload, digest = load_manifest(corpus_path, stem="corpus")
    assert validate_corpus({"schema_version": 1, "payload": payload, "sha256": digest},
                           tokenizer_sha256=tokenizer_files_hash(tokenizer_dir))[1] == digest
    order_run = _run("prepare-order", "--corpus", corpus_path, "--seed", 7,
                     "--updates", 2, "--out-dir", out)
    assert order_run.returncode == 0, order_run.stderr
    state_payload, _ = load_manifest(json.loads(order_run.stdout)["path"], stem="update-order")
    assert state_payload["presentations"] == 128
    assert UpdateOrder.resume([b["id"] for b in payload["partitions"]["training"]["blocks"]],
        {"schema_version": 1, "payload": state_payload,
         "sha256": json.loads(order_run.stdout)["sha256"]}).take_update()
    panel_run = _run("prepare-panels", "--corpus", corpus_path, "--out-dir", out)
    assert panel_run.returncode == 2 and "insufficient" in panel_run.stderr
    config = GPT2Config(vocab_size=10, n_positions=128, n_ctx=128,
                        n_embd=32, n_layer=2, n_head=2)
    config_path = tmp_path / "config.json"
    config_path.write_text(config.to_json_string(), encoding="utf-8")
    init_run = _run("prepare-init", "--config", config_path, "--seed", 7,
                    "--fixture", "--out-dir", out)
    assert init_run.returncode == 0, init_run.stderr
    init_again = _run("prepare-init", "--config", config_path, "--seed", 7,
                      "--fixture", "--out-dir", out)
    assert init_again.returncode == 0 and json.loads(init_again.stdout) == json.loads(init_run.stdout)
    domains = (("sst2", "validation", "sentence"),
               ("gsm8k", "test", "question"),
               ("humaneval", "test", "prompt"))
    domain_args = []
    for name, split, field in domains:
        file = tmp_path / f"{name}.jsonl"
        file.write_text("".join(json.dumps({"split": split, field: f"z {i}",
            "answer": "DO NOT STORE"}) + "\n" for i in range(100)), encoding="utf-8")
        domain_args += [f"--{name}-jsonl", str(file), f"--{name}-revision", "fixture-v1"]
    domain_run = _run("prepare-domains", "--tokenizer-dir", tokenizer_dir,
                      *domain_args, "--out-dir", out)
    assert domain_run.returncode == 0, domain_run.stderr
    domain_payload, _ = load_manifest(json.loads(domain_run.stdout)["path"], stem="xdomain300")
    assert "DO NOT STORE" not in str(domain_payload)
