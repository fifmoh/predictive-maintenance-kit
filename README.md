# Predictive maintenance kit

## Problem and honest boundary
A leakage-safe FD001 remaining-useful-life protocol plus an AI4I transfer study and validation-selected decision economics. Aggregate machine telemetry only; no people monitoring. It consumes exact hash-pinned `researchops==0.1.0`.

## Measured results
None yet. The committed split, window, and 27-cell cost surface are **synthetic fixture evidence**, not measured NASA/UCI performance or savings.

## Demo
Select an engine, inspect prediction context, change illustrative costs, and open transfer-study and technical-note evidence.

## Architecture
See [data lineage](docs/diagrams/data-lineage.mmd).

## Five-minute quickstart
```sh
export RESEARCHOPS_WHEEL=../mlops-research-starter/dist/researchops-0.1.0-py3-none-any.whl
export RESEARCHOPS_SHA256=85df90cf107b3d9c1a2bc128cc8cd19282cd4528b31c501395523fd85c13fe0b
# No sibling clone? Fetch the pinned wheel instead (verify against RESEARCHOPS_SHA256):
# curl -fsSL -o researchops-0.1.0-py3-none-any.whl https://raw.githubusercontent.com/fifmoh/mlops-research-starter/main/dist/researchops-0.1.0-py3-none-any.whl
# export RESEARCHOPS_WHEEL=$PWD/researchops-0.1.0-py3-none-any.whl
make setup leakage-check reproduce
make run
```

## Reproduction
Official downloads require `ACK_DATA_LICENSES=1`. FD001 split identities are fixed before modeling; preprocessing fits only the first 70 engines. AI4I uses a fixed seed-42 stratified split.

## Documentation index
[Security](SECURITY.md) · [Licenses](docs/licenses.md) · [Demo](docs/demo-script.md) · [Technical note](report/technical-note.tex)

## Limitations
Costs are illustrative units, never currency or client savings. Policy costs require complete run-to-failure trajectories and are forbidden on censored official-test engines. Pipeline/decision contracts transfer to AI4I; the RUL target/model do not.

## Dataset/model licenses
NASA and UCI source/license records are in `config/sources.json`; raw datasets and checkpoints are not committed.

## Re-skin guide
Swap only license-approved machine telemetry, sensor names, and cost tuples. Preserve identity splits, train-only preprocessing, validation selection, sealed policy evaluation, and no-people-monitoring boundary.
