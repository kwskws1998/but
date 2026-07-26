# Raw data

Run:

```bash
python cognitive_model_comparsion/download_assets.py --asset all
python cognitive_model_comparsion/download_assets.py --verify-only
```

This installs:

- the two official Provo CSVs;
- the official OneStop Ordinary Paragraph Interest Area ZIP;
- the official SUBTLEX-UK archive/table;
- pinned third-party source archives.

The OneStop ZIP is stored as
`data/raw/onestop/ia_Paragraph_ordinary.csv.zip`. Preparation streams required
columns from its `ia_Paragraph_ordinary.csv` member; it does not extract the
2,455,810,901-byte CSV.

OneStop ZIP provenance:

```text
source: https://osf.io/download/xkgfz/
bytes: 177291322
sha256: 8883478946ee52381e7057683c9e84dc69fcea9054acc34f0c900463a6b546e9
```

All downloaded data and extracted third-party trees are excluded from Git.
Exact sources, sizes, SHA-256 digests, archive member names, and known
licenses are recorded in `asset_manifest.json`.
