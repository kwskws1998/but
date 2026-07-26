# Third-party code

`download_assets.py --asset ob1` downloads the exact OB1–Provo repository
snapshot pinned in `asset_manifest.json`. Use `--asset et2-reference` for the
pinned TorontoCL ET2 training repository and its processed Provo table. The
OB1 wrapper imports the pinned scientific implementation without patching its
equations and supplies only paths, fixed seeds, published parameters, and
structured output aggregation.

The extracted trees and source archives are excluded from Git. Neither pinned
snapshot contains a `LICENSE` file, so this project does not redistribute
those source trees or infer a software license for them.
