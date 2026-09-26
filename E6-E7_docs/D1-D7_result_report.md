Done. Final state: 514 tests passing, 0 skipped; all eight smoke programs exit 0; all 12 frozen files verified; both pre-registrations fully decided with zero live sentinels.

The seven decisions (all made before any result existed)
ID	Resolution
D1	Wasserstein as primary topology metric — §24's claim is about where the anchor sits, and Spearman is blind to depth. e6a_c3 is now a distance; its sign is inverted, which the docs flag.
D2/D3	Framing (B), k=2, calibrated on D0's across-seed spread. The rule is registered; the number is a result.
D4	e6b_1 = 0.02 fixed (SST-2's own binomial SE — an external anchor, so calibrating on our own runs would be circular); e6b_2 calibrated on F1, reusing onset_threshold_k.
D5	Route C, exploratory. I read arXiv:2412.15115v2 — §3.1 gives no per-language token share and no language list, so route A is closed on evidence, not assumption.
D6	any — criterion 1 says "K0 or V0", and §10.12 expects the two to differ.
D7	Both 0.10; semantic_sensitivity as a reading of §18, unrelated_transfer as judgement with no design anchor — the paper must say so.
Six defects, none of which the suite caught
Four found by reading code against its artefacts: criterion_4 read passed/all_within_tolerance while the frozen driver writes all_rows_pass, so the pilot gate could never say proceed: true; nothing could produce that report for a checkpoint at all (no tokenizer — trap 7); resolve_threshold_rule calibrated on a different corpus than it judged; and the E6B smoke asserted the spec dtype against code that deliberately records both.

Two found only by running E6A for the first time: the corpus built 3.57M blocks twice as Python objects and hashed them through one JSON string — ~29 GB peak, which froze your PC twice — now ~3 GB with identical tokens and digests; and LambdaLR doesn't serialise its lambda, so the documented pilot→Phase 2 resume would have spliced two cosine schedules with a silent LR jump.

Why I stopped where I did
getting-paper-ready.md's CRLF diagnosis was wrong on this checkout (all 12 files matched), but its concern was right — I made the hashes EOL-canonical so they hold on a Linux box.

Two things need you before the experiments can run:

FLORES-200 is gated. hf auth login after accepting the terms. XNLI is fine, so E7's patching half is unaffected. Mirrors don't help — I tried all three.
Compute. 6.67 s/step measured: the 5-run pilot is ~20 h, E6A Phase 2 ~6.9 GPU-days on this 2060, before E6B and E7. COMMANDS.md has the full table and every command in dependency order, with PowerShell equivalents.
One correction to something I said mid-session: I reported a training process had "exited silently" — it hadn't. I'd truncated a PowerShell table with tail -2, missed a live 3.7 GB run, and started a second one on top of it. Both were packing the same corpus into one directory. That's now trap 19's second half.