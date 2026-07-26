# Provenance and source audit

## Paper evidence

### GazeReward ET1 training data

The supplied GazeReward PDF states on page 7:

> “This model was trained on the Dundee, GECO, ZuCo1, and ZuCo2 datasets, and predicts total reading time (TRT) per token.”

Source: `2410.01532v3.pdf`, page 7, and
[arXiv 2410.01532](https://arxiv.org/pdf/2410.01532).

This supports using Provo as an external corpus relative to the four named ET1
training corpora. It does not prove that no Provo text was present in any
unreported pretraining resource, so the eventual paper should say “not among
the reported ET1 training corpora,” not make a stronger contamination claim.

### Provo used for the ET2 prediction model

The supplied GazeReward PDF identifies its ET2 architecture as:

> “The second model [Li and Rudzicz, 2021], is based on RoBERTa [Liu et al., 2019] with a regression head on each token.”

Source: `2410.01532v3.pdf`, page 7, and
[arXiv 2410.01532](https://arxiv.org/pdf/2410.01532).

The primary ET2 training paper states:

> “We process the Provo data to be similar to the task data so that they can be combined.”

Source: [Li and Rudzicz (2021), Section 3.4](https://aclanthology.org/2021.cmcl-1.9.pdf).

The paper identifies the corpus as Luke and Christianson (2018), and its
official training repository links `Provo` directly to the same OSF project
used here. The pinned repository's `notebooks/ProvoProcess.py`:

1. maps fixation count, first-fixation duration, regression-path duration,
   and dwell time to `nFix`, `FFD`, `GPT`, and `TRT`;
2. averages these values over participants;
3. calculates `fixProp` from nonzero fixation count; and
4. standardizes every feature before matching the ZuCo target mean and
   standard deviation.

`src/audit_et2_provo.py` applies that code path to the downloaded official OSF
eye-tracking CSV. The reconstruction and the ET2 repository's distributed
`data/provo.csv` have:

| Check | Result |
|---|---:|
| Rows in each table | 2,659 |
| Key rows present in both | 2,659 |
| Left-only/right-only keys | 0 / 0 |
| Maximum TRT absolute difference | `3.55e-15` |
| Maximum difference over all features | `2.70e-13` |
| Audit tolerance | `1e-10` |

This is numerical evidence that the ET2 processed table was produced from the
same official Provo raw release. It is not an alternative raw dataset. It
contains only positions with nonmissing sentence coordinates, excluding 27
raw passage/word positions whose sentence metadata are missing, and its gaze
features are rescaled. Consequently it is unsuitable as the human-millisecond
target for the Human–ET1–OB1 evaluation.

### Why OB1 is the selected cognitive model

The OB1-reader abstract describes one of its key features word for word as:

> “parallel processing of multiple words, modulated by an attentional window of adaptable size”

Source: [Snell et al. (2018), PubMed](https://pubmed.ncbi.nlm.nih.gov/30080066/).

The implementation also uses an adaptive attention width and an asymmetric
attention function in letter coordinates. It does not expose a physical
viewing-distance parameter. The defensible description is therefore:

> OB1 explicitly models an adaptive and asymmetric visuospatial attentional span in letter-based coordinates.

### Exact OB1–Provo experiment

The 2024 OB1–Provo paper states:

> “We use the full cloze completion and reading time data from the Provo corpus [29]. This corpus consists of data from 55 passages (2689 words in total) with an average of 50 words (range: 39–62) and 2.5 sentences (range: 1–5) per passage.”

It also states:

> “We ran 100 simulations per condition in a ‘3x3 + 1’ design: three predictability estimators (cloze, GPT-2 and LLaMA), three predictability weights (low = 0.05, medium = 0.1, and high = 0.2) and a baseline (no predictability).”

And defines TRT as:

> “total reading time, i.e. the sum of fixation durations on the word”

Source: [Lopes Rego et al. (2024), PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012117).

The same paper's data-availability statement says:

> “All the relevant data and source code used to produce the results and analyses presented in this manuscript are available on a Github repository at https://github.com/dritlopes/OB1-reader-model.”

The repository URL now redirects to the snapshot recorded below. Contrary to
the literal data-availability wording, the downloaded snapshot contains only
`.gitignore` placeholders under its data directories. The Provo files and
SUBTLEX-UK resource are absent.

## Downloaded assets

| Asset | Exact source | Local size | SHA-256 | License |
|---|---|---:|---|---|
| Provo eye tracking | [Official OSF file](https://osf.io/download/a32be/) | 69,662,713 B | `38aedcb29bc9171009916eb2bcc2375729f104a2a1005c64a563da94b611b9e7` | CC BY 4.0 |
| Provo predictability norms | [Official OSF file](https://osf.io/download/e4a2m/) | 14,301,138 B | `965fb72eab55f51e08fc1b5622638b85b1085976ff513e2a7bee4adbbd4e6489` | CC BY 4.0 |
| OB1–Provo source archive | [Pinned GitHub commit](https://github.com/dritlopes/language_models_outperform_cloze_predictability_in_a_cognitive_model_of_reading/commit/56b8d6401d1c2c1886a9c6ff9df4a143c6f2c12d) | 55,660 B | `beb9c571c2264f8382fd24a9a5147ed3c7e67a7774d1414e24ba46a5ffb61b1e` | No license file observed |
| ET2 TorontoCL source archive | [Pinned GitHub commit](https://github.com/SPOClab-ca/cmcl-shared-task/commit/19d7af001ab3eab8aa4af02e5a4d11fa204bbedd) | 1,475,665 B | `7048746a5747807fc87e354ba7f902395a2dbe621cc5f26fc394a630e16d44b2` | No license file observed |
| ET2 processed `provo.csv` inside archive | Same pinned archive | 275,651 B | `9bb4c367c8eb95b684065a069e1dd0a21430f5eded0fc20fa6676056cb0e93e4` | No separate license observed |
| SUBTLEX-UK text archive | [University of Nottingham author page](https://psychology.nottingham.ac.uk/subtlex-uk/) | 3,488,497 B | `e40d83af55eb85e9e2bc0d72e1022b10e47e31ff2086ad2edcc18292d6bd6616` | No license statement observed |
| Extracted `SUBTLEX_UK.txt` | Same archive | 18,218,234 B | `9cc02e83efce5c606606578122b792e364dcee52799115218f2cfd596e0fe0f3` | No separate license observed |

Pinned OB1 commit:
`56b8d6401d1c2c1886a9c6ff9df4a143c6f2c12d`.

Pinned ET2 training-code commit:
`19d7af001ab3eab8aa4af02e5a4d11fa204bbedd`.

The Provo OSF node reports “CC-By Attribution 4.0 International.” The OB1
snapshot contains no `LICENSE` file, so the Git repository should retain only
the downloader and provenance, not redistribute the extracted source tree.
The ET2 snapshot also contains no `LICENSE` file and is handled the same way.

### SUBTLEX-UK recovery

The current University of Nottingham author page says word for word:

> “Compressed SUBTLEX-UK files (cleaned, all, bigrams) can be found here.”

Source: [University of Nottingham SUBTLEX-UK page](https://shiny.psychology.nottingham.ac.uk/lpzwjv/SUBTLEX-UK/).

The linked Nottingham host exposes `SUBTLEX-UK.txt.zip`, which is the text
format expected by the pinned OB1 loader. The archive contains exactly one
member, `SUBTLEX-UK.txt`. It was retained untouched, checksum-verified, and
atomically extracted to the underscore filename expected by OB1:
`data/raw/SUBTLEX_UK.txt`.

The extracted table has 160,024 data rows and 27 columns, including
`Spelling` and `LogFreq(Zipf)`. No conversion, alternative frequency norm, or
synthetic replacement was required. The author page does not state a license,
so this repository records no inferred SUBTLEX-UK license and excludes the
download from Git.

## Provo audit

`python cognitive_model_comparsion/src/audit_provo.py` currently reports:

| Check | Observed |
|---|---:|
| Eye-tracking rows | 230,412 |
| Participants | 84 |
| Passages | 55 |
| Raw eye-tracking text/word positions | 2,685 |
| Predictability rows | 41,326 |
| Raw predictability text/word positions | 2,687 |
| Published-correction Human–OB1 positions | 2,686 |

The paper-level “2,689 words” is a corpus description, not the final inner
join count produced by the downloadable CSVs and published OB1 corrections.
The public data contain:

- a duplicated raw position at passage 18, word 3;
- the `evolution` position correction used by OB1;
- index shifts in passages 3 and 13;
- the malformed `Ñ` token in passage 36;
- three eye-tracking omissions in passage 55;
- 4,788 eye-tracking rows with no word number.

After reproducing the position corrections and exclusions encoded in the
published OB1 evaluator, the human and model grids match exactly at 2,686
positions across all 55 passages. The experiment must save both the raw audit
and the corrected alignment table.

## Upstream OB1 reproducibility findings

The pinned code could not be executed reproducibly as published without a
local wrapper:

1. `src/main.py` hardcodes `useparser=False`, so the documented CLI and
   `run_simulations.sh` arguments are ignored.
2. `src/pre_process_stimuli_file.py` reads from one directory and writes to a
   different directory than the repository README specifies.
3. Provo raw/processed data and frequency maps are absent.
4. `src/utils.py` requires `../data/raw/SUBTLEX_UK.txt`, which is absent from
   the source snapshot.
5. The LLaMA path references the obsolete
   `decapoda-research/llama-7b-hf` identifier and manual tokenizer edits.
6. The checked-in `environment.yml` contains a developer-specific absolute
   prefix and should not be used as a portable lock file.
7. The source snapshot has no explicit software license.

The implemented adapter preserves the pinned scientific equations and
published baseline parameter values. It supplies only an isolated path
layout, official Provo/SUBTLEX inputs, deterministic seeding, structured
fixation output, and word-grid aggregation.

## Executed implementation audit

The local reproducibility gates produced the following observations:

| Gate | Observation |
|---|---:|
| Frozen ET1 checkpoint | 70,144,951 B; repository-pinned SHA-256 validated |
| Frozen ET1 passages | 55 |
| ET1 token rows | 3,715 |
| Unassigned non-special ET1 tokens | 0 |
| ET1 canonical word rows | 2,686 |
| OB1 same-seed separate-process repeat | exact |
| OB1 seed-0 / seed-1 one-passage fixation rows | 58 / 57 |
| OB1 full 55-passage, one-reader runtime | 280.18 s |
| OB1 full fixation rows | 2,783 |
| OB1 word rows / zero-TVT rows | 2,686 / 388 |
| OB1 regression fixations retained | 619 |
| Symmetric/asymmetric mass checks in integration smoke | 110 / 110 passed |

The 110 redistribution checks used synthetic widths solely to validate the
execution path; they are not a result condition. Actual learned
redistribution requires the reported OASST1 checkpoint state files. The
55-passage OB1 run used one virtual reader and is a pre-run gate, not the
paper's 100-reader condition.
