# D19: RTX 4080 SUPER class transfer

Researcher approval date: 2026-09-28. The sealed authority is
[`protocols/s1_researcher_amendment_d19_4080_class_20260928.json`](../../protocols/s1_researcher_amendment_d19_4080_class_20260928.json).
The D01–D18 register is byte-preserved because its original approval hashes that file.

NodiPC is the measured reference `NVIDIA GeForce RTX 4080 SUPER`, UUID
`GPU-72b4b307-b613-c35e-ea32-53f4431de9ee`. Its approved production-shape
qualification covers C0/C1/C2/C5/C6. The researcher authorizes fresh jobs for
those conditions on equivalent exact-model RTX 4080 SUPER hosts, including
NaveedPC and Adrita-PC, without another per-card VRAM-headroom pass.

NaveedPC's observed C2/C5 microbatch-4 cycles completed but missed the former
free-VRAM threshold by 96,025,804 and 228,498,636 bytes. Those measurements
remain failed per-card engineering evidence. Class transfer is a researcher
waiver that accepts Windows/WDDM free-memory variation as operational risk.
An actual production OOM aborts and preserves evidence; microbatch 4,
accumulation 16, sequence length, optimizer and schedule do not retune.

An exact model, locked software/numerics, artifacts, source and protocol remain
mandatory. Every fresh run records its physical UUID in immutable run and
checkpoint identity. Resume requires that same UUID, so no mid-run migration is
authorized. C3 has no reference 4080 approval and C4/REL remains prohibited
on 4080. RTX 3090 assignments, AbdullahPC's exact UUID, and C1/C2 bridges
remain unchanged. The predecessor production root remains the historical root
for work already started under it; the successor is prospective only.
