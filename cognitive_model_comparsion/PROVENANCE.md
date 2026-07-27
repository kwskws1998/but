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

### GazeReward ET1 token coordinate

The supplied GazeReward PDF labels the relevant Appendix table:

> “Example of mapping TRT between two different tokenizers. TRT (1) represents the process used for the first ET predictor, and TRT (2) for the second ET predictor.”

Source: `2410.01532v3.pdf`, Table 8, and
[arXiv 2410.01532](https://arxiv.org/pdf/2410.01532).

Among the 12 supplied runs, the checkpoint with the highest reported original
reward-model accuracy (`0.76942`) has `sigma_left=0.3738` and
`sigma_right=3.21289`. These values came from the ET1 + GazeConcat, TRT-only
condition and were selected without using Provo or OB1 results. An earlier
`0.41553/3.46115` pair remains in the historical completed OB1 cache directory
and integration-smoke commands but is not the selected rebuttal checkpoint.
Consequently the primary redistribution is applied on the native ET1/T5 token
sequence, with the same production attention-mask semantics used during OASST1
training. Only after redistribution are non-special T5 tokens associated to
corpus words by exact character offsets and summed:

\[
E_i=\sum_{t\in word(i)}TRT_t^{ET1}.
\]

This word sum is the common external-evaluation coordinate, not a refit or a
conversion of sigma to word, letter, or visual-angle units. The
special-token-excluded run changes only the redistribution mask; it does not
change this aggregation.

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

### Attention-profile analysis contract

The OB1 paper defines the directional parameter word for word:

> “Asym is equal to 1 toward the right and 0.25 toward the left”

It also states:

> “Outside the Gaussian, the attentional weight is set to constant”

Source: Snell et al. (2018), p. 973, in the
[OB1 paper](https://research.vu.nl/ws/portalfiles/portal/72578613/OB1_reader_A_model_of_word_recognition_and_eye_movements_in_text_reading.pdf).

The vendored implementation adds the constant `0.25` to the asymmetric Gaussian
in `src/reading_helper_functions.py`. The attention-profile analysis therefore
defines:

- `focused`: the fixation-onset asymmetric Gaussian component without the
  constant residual;
- `full`: the same focused component plus the constant `0.25`.

Neither variant reconstructs acuity, within-fixation attention shifts, lexical
activation, saccade control, or final TVT. The saved trajectory manifest records
`attention_skew=3`; `skew=4` re-evaluates the attention equation at the saved
fixation locations and does not rerun OB1. Letter-coordinate attention is
evaluated at T5 token centers and expressed as relative native T5 token offsets.

The `raw_delta` machine ID in this analysis is a no-redistribution unit impulse:
all allocation weight remains at the source token. It is not behavior-level
`ET1 raw`, and actual ET1-predicted TRT magnitudes are not used. The ET1 output
table supplies only native T5 token geometry and character alignment.

The primary candidate-support policy is `fixation_matched`. It evaluates and
normalizes each candidate on the exact relative-token offsets visible at each
saved fixation, matching the support used for the OB1 component, and then pools
both with identical fixation weights. The earlier implementation was hybrid:
OB1 was normalized within each fixation window, but candidates were normalized
once on the global union of observed offsets. The `global` policy preserves
that calculation only as a labeled legacy sensitivity.

### Exact OB1–Provo experiment

The 2024 OB1–Provo paper reports “55 passages (2689 words in total),” states
“We ran 100 simulations per condition,” and defines TRT as “the sum of fixation
durations on the word.”

Source: [Lopes Rego et al. (2024), PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012117).

The same paper's data-availability statement points to the authors' GitHub
repository for the relevant data and source code.

The repository URL now redirects to the snapshot recorded below. Contrary to
the literal data-availability wording, the downloaded snapshot contains only
`.gitignore` placeholders under its data directories. The Provo files and
SUBTLEX-UK resource are absent.

### OneStop external-validation corpus

The OneStop data descriptor reports the corpus structure:

> “with a total of 162 paragraphs across the 30 articles.”

For the released word coordinate, it states:

> “Interest Areas in the Data Viewer reports span whitespace tokenized words.”

Sources:
[Berzak et al. (2025), Scientific Data](https://www.nature.com/articles/s41597-025-06272-2)
and the
[official variable documentation](https://lacclab.github.io/OneStop-Eye-Movements/variables).

OneStop contains 360 participants across two equally assigned reading
regimes. This experiment deliberately uses only the official
[OneStop Ordinary Reading](https://osf.io/zn9sq/) Paragraph Interest Area ZIP,
then selects non-practice, non-repeated, Advanced paragraphs. The resulting
live archive audit contains 180 Ordinary/Gathering participants, 30 articles,
162 paragraphs, and 19,440 exact whitespace-token word positions. These are
observed release counts, not counts inferred solely from the paper.

The official known-issues page states:

> “In some cases this resulted in 11 lines of text instead of 10.”

It also warns:

> “this results in two different versions of text.”

Source:
[OneStop Known Issues](https://lacclab.github.io/OneStop-Eye-Movements/known_issues.html).

This is why preparation does not silently collapse exact text variants. It
selects the variant read by the most participants for each base paragraph,
uses deterministic lexical and SHA-256 tie-breaks, and saves every selected
and excluded variant.

The official repository states:

> “The eye tracking data, code, and anonymized participant questionnaire responses are released under a Creative Commons Attribution 4.0 International License.”

Source:
[OneStop official repository](https://github.com/lacclab/OneStop-Eye-Movements#license).
The same license section assigns the underlying text and auxiliary OneStopQA
annotations to CC BY-SA 4.0.

## Downloaded assets

| Asset | Exact source | Local size | SHA-256 | License |
|---|---|---:|---|---|
| Provo eye tracking | [Official OSF file](https://osf.io/download/a32be/) | 69,662,713 B | `38aedcb29bc9171009916eb2bcc2375729f104a2a1005c64a563da94b611b9e7` | CC BY 4.0 |
| Provo predictability norms | [Official OSF file](https://osf.io/download/e4a2m/) | 14,301,138 B | `965fb72eab55f51e08fc1b5622638b85b1085976ff513e2a7bee4adbbd4e6489` | CC BY 4.0 |
| OneStop Ordinary Paragraph IA ZIP | [Official OSF file](https://osf.io/download/xkgfz/) | 177,291,322 B | `8883478946ee52381e7057683c9e84dc69fcea9054acc34f0c900463a6b546e9` | Eye data CC BY 4.0; text/annotations CC BY-SA 4.0 |
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

## OneStop archive and preparation audit

The official `ia_Paragraph_ordinary.csv.zip` is retained compressed. Its
single data member, `ia_Paragraph_ordinary.csv`, is 2,455,810,901 bytes when
uncompressed. Preparation reads only required columns in 50,000-row chunks
directly from the ZIP; it never writes the expanded CSV.

The strict preparation audit currently records:

| Check | Observed |
|---|---:|
| Raw Interest Area rows | 1,104,883 |
| Advanced, non-practice Ordinary rows | 583,051 |
| Ordinary/Gathering participants | 180 |
| Article clusters | 30 |
| Base paragraphs | 162 |
| Participant-paragraph trials before variant selection | 4,859 |
| Participant-paragraph trials retained | 4,759 |
| Minority-variant trials excluded | 100 |
| Paragraphs with multiple exact text variants | 11 |
| Canonical whitespace words | 19,440 |
| Retained participant-word rows | 570,641 |
| Reader-count range per selected paragraph | 17–30 |
| Punctuation-only words incompatible with OB1 normalization | 95 |
| Paragraphs with zero OB1 transformations | 107 |
| Article clusters represented by clean paragraphs | 28 |
| Paragraphs containing at least one transformed position | 55 |

Variant selection uses participant count only; it never reads Human TRT or
model output. The 19,440-word Human/ET1 grid remains intact. The 95
punctuation-only words are marked `ob1_evaluable=False`, recorded in
`ob1_token_transformations.csv`, and excluded from every method on the common
Human–ET1–OB1 metric grid.

Although common-grid exclusion prevents a direct metric at those positions,
an OB1 surrogate used while simulating the paragraph could influence
neighboring fixations. The optional clean-passage sensitivity therefore
selects the 107 paragraphs with
`ob1_incompatible_words_excluded == 0`, excludes all 55 affected paragraphs,
and writes separate results under `ob1_clean_passages/`. The retained
paragraphs span 28 resampling clusters. This does not alter the 162-paragraph
primary analysis.

`IA_DWELL_TIME` is preserved as released. The unconditional target includes
zero values. The conditional target averages positive values only; when no
retained reader has positive dwell on a word, the saved conditional value is
missing and that word is excluded only from the conditional metric.

Paragraphs from the same OneStop article are not independent experimental
units. The evaluator therefore uses the 30 article IDs as resampling clusters
for percentile confidence intervals and paired sign-flip tests.

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
execution path; they are not a result condition. Actual learned redistribution
requires either verified effective sigma values or the corresponding OASST1
checkpoint state files. The 55-passage OB1 run used one virtual reader and is a
pre-run gate, not the paper's 100-simulation condition.

The rebuttal-grade cached post-processing must additionally record the SHA-256
hash of `ob1_fixations.csv`, its simulation/seed/passage counts, the trajectory
manifest hash and condition, the selected profile component, and whether the
requested attention-function skew matches or reweights the saved trajectory.
It must also record `candidate_support_policy`; only `fixation_matched` is
primary, while `global` is a legacy support sensitivity. The updated
`compare-attention-profile` audit writes these fields; they must be read from
the completed 100-simulation server output rather than inferred from the local
one-reader gate.

The previously printed selected-s11 Spearman values `0.720` for `skew=3` and
`0.737` for `skew=4` were generated by the legacy global-support calculation.
They cannot be cited as primary after the support correction. The completed
100-simulation cache must be post-processed again with
`--candidate-support-policy fixation_matched`; no corrected 100-simulation
values are asserted in this provenance record before that rerun is inspected.

The OneStop official archive, strict 162-paragraph canonical build, exact
variant audit, and OB1 compatibility audit have been executed. The complete
100-seed OneStop OB1 simulation and its joined scientific metric tables have
not been executed in this checkout and must not be reported as results.
