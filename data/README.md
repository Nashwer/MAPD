# Data layout

- `nq/`: converted Natural Questions train/evaluation records
- `hotpotqa/`: converted HotpotQA train/evaluation records
- `protocols/`: accepted offline protocol artifacts
- `wiki18/index/`: local wiki-18 retrieval index

Large datasets and index files are ignored by Git; only this directory contract is versioned.

Current status (2026-09-24): repository fixtures are sufficient for local and
GPU smoke tests, but the full NQ/HotpotQA corpus and wiki-18 index have not yet
been downloaded or normalized. Their presence must not be inferred from the
directory names.
