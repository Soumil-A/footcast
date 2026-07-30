# Data

Raw and processed datasets are generated locally and are not committed.

- `download_manifest.json`: reviewed source URLs, checksums, season bounds, and
  expected dimensions
- `raw/`: unchanged, checksum-verified Football-Data CSV files
- `processed/matches.csv`: reproducibly generated canonical match table
- `processed/pre_match_features.csv`: development-only Phase 3 feature table

Run the complete Phase 1 workflow from the repository root:

```bash
python -m footcast.data.pipeline
```

The command downloads only absent raw files. It will never overwrite an
existing raw file whose checksum differs from the manifest. If Football-Data
revises a historical file, inspect the upstream change and its quality report
before intentionally updating the manifest checksum.

The processed table contains only the stable match identity, date, score, and
result fields needed to establish a trustworthy base dataset. Match statistics
and odds remain visible in the source-schema audit but are not transformed into
features in Phase 1.

After Phase 1 data exists, build the development feature table with:

```bash
python -m footcast.features.build_features
```

The generated CSV remains local and ignored by Git. Its code, feature contract,
quality report, and hand-calculated tests are versioned.
