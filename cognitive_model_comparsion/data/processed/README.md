# Processed data

`python cognitive_model_comparsion/main.py prepare-provo` generates:

- `provo_passages.csv`;
- `provo_words.csv`;
- `provo_excluded_positions.csv`;
- `provo_prepare_audit.json`.

The tables retain raw and corrected coordinates, exact character offsets,
Human TRT with and without skips, reader/skip counts, known string exceptions,
and every exclusion reason. Generated files are excluded from Git and are
reproducible from the checksum-verified raw assets.
