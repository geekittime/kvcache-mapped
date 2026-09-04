# PRE-FORMAL protocol and claim boundary

1. Read-only bind the pairwise-v3 graph `PUBLISHED.json`, graph deep receipt,
   and allocation-v3 infeasibility publication by absolute path, byte count,
   SHA-256, sidecar, and completion marker.  An allocation-v3 deep receipt is
   a required external gate; its absent/in-progress state is represented by a
   placeholder and never synthesized.
2. Validate records and hyperedges against the exact v3 privacy-safe schemas.
   The source graph report's thresholds and direct-collision policy are copied
   and hash-checked; no collision relation is recomputed or changed.
3. Run A/B/C only after the external gate is complete.  The public target
   order is immutable and all 18 upper bounds remain the v3 quotas.
4. Run D in two fresh single-thread processes with paired Python/HiGHS seeds.
   Every solver result is typed, finite where required, status-0 Optimal, and
   independently checked against the sparse matrix.  Selection JSONL bytes and
   semantically relevant reports must agree across replicas.
5. A separate deep process validates the quota and selection publications and
   binds the external allocation-v3 deep receipt.  A crash leaves
   `IN_PROGRESS.json`; only a final atomic marker is consumable.

This PRE-FORMAL revision intentionally stops after binding the completed
allocation-v3 deep receipt and rerunning the answer-free CPU regression.  It
does not start A/B/C/D, a formal registry, either model, a GPU, or any answer /
support read, and it makes no superiority/accuracy claim.  Every executable
entrypoint must receive `PYTHONDONTWRITEBYTECODE=1` and an absolute,
initially-absent `PYTHONPYCACHEPREFIX`; an existing or implicit cache is a
fail-closed error.
