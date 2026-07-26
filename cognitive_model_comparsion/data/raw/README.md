# Raw data

Run:

```bash
python cognitive_model_comparsion/download_assets.py --asset all
python cognitive_model_comparsion/download_assets.py --verify-only
```

This installs the two official Provo CSVs and the official SUBTLEX-UK text
archive/table. All are excluded from Git. Exact sources, sizes, SHA-256
digests, archive member names, and known licenses are recorded in
`asset_manifest.json`.
