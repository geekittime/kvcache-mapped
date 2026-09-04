# Method note: a conservative, dataset-blind gate

## Motivation

The completed old-pilot comparison showed a split failure mode.  Shrinkage was
roughly neutral on HotpotQA but useful on MuSiQue, while the inexpensive signals
available before a second decode (prompt length and the length of the first
generation) were correlated with the latter workload.  Selecting on either
signal alone is too permissive for a Hotpot-safe claim.  This attempt therefore
uses an intersection gate.  The gate is not allowed to infer a dataset name;
the correlation is only a hypothesis to be tested on a fresh registry.

## Serving rule

Let (p) be the number of tokens in the complete prompt and (g) be the number
of tokens emitted by the completed NVIDIA/R1 pass.  Define

\[
  z = 1\{p \ge 2800\}\,1\{g \ge 16\}.
\]

For a valid, complete first pass, (z=1) reserves one unit of the declared
batch budget and runs `shrinkage_r1`; otherwise the request uses the NVIDIA k20
mapper.  The budget is

\[
 B(n)=\min(32,\lfloor0.30n\rfloor),
\]

where (n) is frozen before routing.  Candidates beyond (B(n)) are not
reconsidered in a later batch.  Original input order is the fixed tie break.

The implementation has no argument for a dataset/task label, output string,
reference answer, score, wall-clock measurement, cap flag, or GPU statistic.
This is an intentional information-flow boundary, not merely a convention in
the analysis script.

## Three arms

Every fresh evaluation should collect three paired arms on the same prompt:

1. `native14`: independent target-model reference/control;
2. `nvidia_k20`: the immutable NVIDIA mapper baseline;
3. `shrinkage_r1`: the candidate mapper.

The router policy is compared with always-`nvidia_k20`; `native14` is reported
as a quality/latency reference and is never used to choose a route.  An
always-shrinkage arm is useful as a diagnostic but is not the proposed policy.

## Why the intersection is conservative

An OR gate can spend its budget on many ordinary Hotpot requests.  An AND gate
has lower recall but limits escalation to requests exhibiting two independent,
observable signs.  Since the old pilot has zero Hotpot rows satisfying both
thresholds, no Hotpot benefit is claimed here.  The fresh registry is the only
place where the trade-off can be estimated.

## Cost accounting

For each arm record generated-token count, stop reason, cap status, wall time,
and peak memory.  These fields are analysis outputs and are never routed on.
Report accuracy and cost separately for the pure mapper pass and any optional
target replay.  A router that saves tokens but violates the Hotpot safety gate
is a failure, not a success.
