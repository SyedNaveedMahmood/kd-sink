WP12 is complete and verified. Summary of what happened:

The delivered files
files (7).zip held four files, all relocated: 08_AGGREGATOR_EXTENSION_SPEC.md and DECISIONS_REQUIRED.md → new_design_plans/, and the v3 pre-registrations → their configs/ directories. §2.5 and §2.6 of NEXT_STEPS.md are now closed — design §8.7, §10.11, §10.12 and §18 are transcribed.

I made one edit to the delivered YAMLs: a reason: line on each PENDING_DECISION_* entry, because both aggregators and test_aggregate_contracts.py read that key and the artefact would otherwise report a pending criterion with an empty explanation. No threshold, tier or number was touched.

What I built (WP12)
aggregate_transformation.py — kind: corpus_pair, absolute_difference, single-arm entries, compound: {all_of}, matched_loss_reference, seedless P8M pairing, seed_consistency: all, e6b_go_no_go with combine: all, threshold_rule, and a pending-decision sweep.
aggregate_crosslingual.py — kind: source_level, four new observed_pattern keys, retrieval threaded into the matrix through a reduction shared with criterion e7_1, claim_gate_component, go_no_go_combine, and criterion_3_combine.
evaluate_transformation.py — --evaluate-public-reference, usable with or without a run directory.
Nine new test files, 87 tests.
Verification (all run, all green)
pytest tests/ -q → 372 passed, 0 skipped (was 285).
Smoke runner → 7/7 exit 0.
Real GPU run on the RTX 2060: roneneldan/TinyStories-8M against the real TinyStories-33M teacher — condition="P8M", step -1, band [2,8), fingerprint_cosine_to_teacher 0.2113/0.4880, and with ΔCE, validation_ce 1.8979 and functional_cosine_to_teacher 0.9683.
The v3 §10.12 matrix passes the exhaustive 64-assignment disjointness sweep.
No frozen file touched.
Four things you should know
08 §12's "test_e7_aggregate_contracts.py passes unchanged" could not hold. One of its 21 tests asserted v1 sentinel identifiers (e7_3, e7_4, PENDING_DESIGN_10_12) that v3 renames. Two assertions in that one test were updated; the other 20 and the three other named files pass unchanged.
08 §5.1's worked example is arithmetically wrong — [0.30, 0.05, 0.05] averages to 0.1333, below its own 0.15 threshold, so it isn't met under either reading. I asserted the literal triple as it behaves and made the intended contrast with [0.30, 0.10, 0.10].
I narrowed the pending sweep after it broke two existing tests. A blanket "any PENDING blocks the verdict" let an undecided interpretation matrix veto a computed result. Sentinels now carry blocks_decision; only criterion_3_combine and criterion-consumed contrasts block. Rationale is in §1.4.
The seven decisions D1–D7 are untouched, as you asked. Both verdicts read incomplete. §2.7 lists them; D5 (language tiers) is the one to do first — it may need the Qwen2.5 report, and it's the easiest place to accidentally tier on the outcome.
WP6 (E6B) remains the only work package left, still conditional.