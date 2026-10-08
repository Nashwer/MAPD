# Data layout

- `nq/`: converted Natural Questions train/evaluation records
- `hotpotqa/`: converted HotpotQA train/evaluation records
- `protocols/`: accepted offline protocol artifacts
- `wiki18/index/`: local wiki-18 retrieval index
- `downloads/`: versioned source Parquet/corpus downloads
- `training/`: deterministic 25,600-example training subset and manifest
- `evaluation/`: normalized held-out records used for leakage exclusion

Large datasets and index files are ignored by Git; only this directory contract is versioned.

Run `bash mapd.sh data-setup` to populate the QA directories and
`bash mapd.sh wiki-setup` to build `wiki18/index/wiki18.sqlite3`. Repository
fixtures are sufficient for local tests, but full data/index presence must not
be inferred from directory names. Generated files remain ignored by Git.

Run `bash mapd.sh data-export` to validate and package the prepared 25,600-example
training reconstruction plus its held-out split and provenance manifests. Add
the argument `source` only when the original source Parquet files must also be
carried to another machine. Neither form includes wiki-18 or generated protocols.
