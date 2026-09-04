# Method note

Let `U_t` be the 18 original pairwise-v3 quotas and `q_t` the repaired integer
quotas.  Binary variables `y_{t,r}` assign an eligible privacy-safe record to
at most one target.  Every published direct hyperedge `E` contributes exactly
the v3 constraint

```
sum_{t,r in E} y_{t,r} <= 1.
```

No transitive closure is introduced.  The record assignment, upper quota,
screen/confirm ratio, and 4-hop buffer are all linear constraints.

Stage A adds a continuous `lambda in [0,1]` and constraints
`sum_r y_{t,r} >= U_t lambda`; it maximizes lambda only to locate the relevant
finite breakpoint.  Since integer quotas imply
`lambda* in {k/U_t}`, the implementation then performs an exact
`fractions.Fraction` breakpoint binary search.  A candidate is feasible only
when a status-0/Optimal HiGHS solve returns a finite binary incumbent satisfying
the complete matrix.  A status-2 result is accepted only with `success=false`,
`x=null`, and the pinned HiGHS “The problem is infeasible … model_status is
Infeasible” semantics.  Any limit, numerical, unknown, malformed, or
non-finite result fails closed.

Stage B adds `q_t >= ceil(U_t lambda*)` and minimizes `-sum_t q_t`.  Stage C
adds the B total and, one target at a time, minimizes `-q_t` while fixing all
earlier choices.  The resulting vector is the lexicographically largest vector
under the public target order; it is independent of arbitrary MILP incumbent
choice.  Stage D rebuilds the fixed-quota matrix and uses the frozen SHA-256
cost formula.  The cost numerator is the first 13 hexadecimal digits of
`sha256(namespace + NUL + target_index + NUL + source_projection_sha256)`;
the two seed/PID replicas must produce byte-identical JSONL.

All phase reports carry typed status/success/message, finite objectives where
applicable, PID and seed evidence, and explicit privacy flags.  The framework
does not open raw train JSON, answer fields, locked metrics, or model output.

