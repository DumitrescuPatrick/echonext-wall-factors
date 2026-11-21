# EchoNext Wall-Thickness Factors — Code Only

Source-only companion to the exploratory wall-thickness factor study. This package deliberately excludes the manuscript, figures, aggregate results, patient-level data and trained artifacts. No study result is reproduced merely by installing or importing this code.

## Included

- `src/`: original Python models, waveform handling, factor extraction, readouts, statistical evaluation and supporting code.
- `config/`: original protocol/model configurations and dependency references. Settings and schema names describe methods; they are not patient records.
- `tests/statistics/`: original tests using explicitly artificial arrays/rows.
- `tests/release/`, `tools/`, `audit/`: source-only release privacy, integrity and structure checks.
- dependency files and `.gitignore`.

No dataset, patient ID values, recordings, waveform paths, clinical dates, linked ECG–echo records, embeddings, individual predictions, weights, PCA artifacts, caches, credentials, saved result tables/JSON, publication figures or manuscript files are supplied. Schema fields such as `subject_id` in code are variable names only. Synthetic fixture years and arrays are invented for software tests, not extracted from patients.

## Install and run software checks

Use Python 3.12 and an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python tools/verify_code_only.py
python -m unittest discover -s tests/release -v
PYTHONPATH=src python -m unittest discover -s tests/statistics -v
```

These commands verify the delivered release and run synthetic tests. They do not download data, train a network, access clinical records or recompute clinical results. Dependency pins describe the validated local preparation environment, not a newly recovered original training environment. `requirements-render.txt` supplies numerical/plotting libraries used by source modules; no rendered files or data are included.

## Clinical execution prerequisites

The analytical code retains its original private-layout assumptions. Full clinical reproduction requires separately authorized source data, exact prior preprocessing/splits and private frozen checkpoints/PCA transformations/caches. Those artifacts are absent by design. This is not a turnkey data download pipeline, pretrained service or self-contained clinical reproduction. Do not run historical training or study launchers merely to test this repository. Importing the tested modules does not launch their guarded CLI routines.

MIMIC and EchoNext have provider-specific access agreements. Obtain access directly from the relevant provider and keep restricted artifacts locally; do not add them to GitHub. The local release's hash allowlist and .gitignore are safeguards, not permission to redistribute data or a guarantee about arbitrary future additions. This ZIP contains no Git history; it cannot erase earlier online commits.

## Publication and license

The scientific study remains exploratory; this source release adds no clinical experiment or independent validation. Manuscript versions and research outcomes remain in separate packages. Authors must select/approve a code license and citation metadata; neither a license grant nor author/ethics/funding/conflict/AI-use declaration is invented here. No GitHub repository was created or published by this preparation.

Upload only the extracted code-only folder contents, including .gitignore. Preserve earlier frozen study materials separately.
