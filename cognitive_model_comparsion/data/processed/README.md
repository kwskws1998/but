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

`python cognitive_model_comparsion/main.py prepare-onestop --strict`
generates under `data/processed/onestop/`:

- `onestop_passages.csv`;
- `onestop_words.csv`;
- `onestop_participant_words.csv`;
- `onestop_variant_audit.csv`;
- `onestop_prepare_audit.json`.

The strict official-archive audit requires 180 Ordinary/Gathering
participants, 30 article clusters, 162 Advanced paragraphs, 19,440 canonical
whitespace words, 11 multi-variant paragraphs, 100 excluded minority-variant
trials, and 95 punctuation-only words marked `ob1_evaluable=False`.

The canonical Human grid retains all 19,440 words. OB1 incompatibility is an
explicit common-grid exclusion at evaluation time, not silent deletion during
preparation. The passage audit also identifies 107 paragraphs with zero OB1
transformation, spanning 28 article clusters, and 55 affected paragraphs for
the optional clean-passage sensitivity.
