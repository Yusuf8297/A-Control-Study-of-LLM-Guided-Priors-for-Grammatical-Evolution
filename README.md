# Feature selection for Grammatical Evolution in the few-shot regime

Can a data-free knowledge prior obtained from a large language model hold
its own against classical feature selection as the source of the feature
subset handed to an interpretable Grammatical Evolution (GE) learner that
trains on only k labelled rows, and when does the answer change: with the
labelling budget k, with the task (classification or regression), and with
whether the feature names carry any meaning at all?

The experiment compares, on identical data draws, four data-driven
selectors (correlation, mutual information, PCA loadings and random-forest
importance), the LLM prior, a size-K random subset that is the prior's
semantic floor, and an all-features baseline, across 24 datasets: the
eleven of the FeatLLM few-shot benchmark (Han et al., ICML 2024) and the
thirteen that complete the eighteen-dataset benchmark of Memar, Wang and
Agah, "When Do LLMs Help with Feature Selection? A Systematic Empirical
Study" (arXiv 2610.28556, October 2026). Nineteen datasets are
classification tasks scored by test ROC-AUC and five are regression tasks
scored by the test Spearman correlation of the evolved rule's output, with a
calibrated R² as a secondary metric.

## How this study differs from Memar et al. (2026)

Memar et al. compare frozen and RL-adapted LLM selectors with classical
selectors under a fixed XGBoost learner trained on 60% of each dataset,
report ROC-AUC and R² descriptively over five seeded partitions, and
isolate feature-name semantics with a cardinality-matched
original/anonymised/shuffled diagnostic. Their conclusion is that
classical selectors win when data are plentiful. This study asks the
complementary question and answers it with a different design:

1. The labelling budget is the independent variable. Every method is
   evaluated at k = 4, 8, 16, 32 and 64 labelled rows and at a capped
   full level, so the comparison traces where, if anywhere, a data-free
   prior crosses the data-driven selectors as data become scarce. Memar et
   al. have one, data-rich, operating point.
2. The learner is an interpretable symbolic model, not a gradient-boosted
   ensemble. Selection enters GE through the grammar's terminal rule, so
   the subset shapes the search space itself; the evolved rule is readable.
3. Every method receives the same budget K on every dataset, so
   cardinality is matched by construction rather than by a separate
   control; the data-driven selectors are run under two information
   regimes (whole training partition, or the same k rows as the learner),
   each as its own campaign.
4. The semantic mechanism is tested two ways that need no extra LLM calls:
   a size-K random subset at matched cardinality is exactly what a
   data-free prior reduces to under anonymised names (semantic floor), and
   four datasets whose feature names are indices (ionosphere, optical
   digits, geographical music, sonar) form a pre-specified anonymous
   stratum against the twenty named ones (semantic-anchor moderator).
5. The analysis is pre-registered and confirmatory in the Demšar sense:
   blocked on dataset, Wilcoxon signed-rank with Holm correction and
   Cliff's delta, Friedman with the Nemenyi post hoc, ten draws per cell,
   medians as the primary summary, null results reported as findings.
6. The prior is a cached response to feature names and descriptions
   alone; it never sees rows, labels or summary statistics (Memar et al.'s
   prompts carry per-feature summary statistics).

The two studies share six datasets; `results/<design>/analysis/contact_memar_table1.csv`
places our numbers beside their Table I on those, as a point of contact
between protocols, not as a comparison of methods.

## Protocol

Splits, grammar, search and budgets follow the FeatLLM few-shot protocol
(Han et al., ICML 2024): a 20% stratified test split, a class-balanced draw
of k labelled rows for the learner with k in {4, 8, 16, 32, 64} plus a
capped full-data level (training rows capped at 1000), test ROC-AUC as the
metric, ten draws per cell. Regression tasks stratify the split and the
draw on target quartiles and score test R². A selection method changes
only the grammar's terminal rule, restricting it to the chosen features at
the dataset's fixed budget K; every other rule, and the search itself, is
identical across methods. The test partition reaches no selector and no
learner before scoring. A shot level runs on a multiclass dataset only
when every class can receive a labelled row, so dermatology and mice
protein start at k = 8 and optical digits at k = 16; every table carries
the number of datasets N at each level.

| Method | Subset source | Sees labels | Sees rows |
|---|---|---|---|
| All features | no selection (baseline) | - | - |
| Correlation | binary: point-biserial \|r\|; multiclass: correlation ratio; regression: \|Pearson r\| | yes | yes |
| Mutual information | kNN estimator (classification or regression) | yes | yes |
| PCA loadings | explained-variance-weighted absolute loadings | no | yes |
| RF importance | random-forest impurity importance, 100 trees (classifier or regressor) | yes | yes |
| LLM prior | cached LLM response from names, descriptions and task text | no | no |
| Random subset | size-K subset drawn at random per draw seed; the semantic floor | no | no |

Each data-driven selector scores standardised rows: the whole training
partition of the draw's split under the full-information design, or only
the k labelled rows under the matched design. Scores are computed per
encoded column, aggregated to original features by group maximum, ranked
with ties broken towards column order, and truncated at K. Under the
full-information design a selector's subset is fixed across the shot
levels of a draw, as is the LLM subset; any trend with k in those results
is the learner's.

The GE learner (GRAPE; population 200, 40 generations, one search per
binary task or per class one-versus-rest) is evolved on the k rows alone.
Both fitness functions are rank-based, so the evolved expression's scale
never has to be calibrated on k rows: classification fitness is
1 − training ROC-AUC, regression fitness is 1 − training Spearman ρ, and
the regression test metric is the Spearman ρ between the rule's output and
the target. A calibrated R² is logged alongside as the secondary regression
metric (`test_r2`): a linear rescaling a·s + b fitted by least squares on
the k rows (Keijzer 2003), applied to test values clipped to the training
range of s, floored at −1 (a predictor twice as bad as the mean) so one
degenerate rule cannot dominate a per-dataset mean, identically for every
method; floored cells are counted in `r2_floor_counts.csv`. The secondary
metric is the one Memar et al. report, and the reason it is secondary is
recorded as change 6 below. GRAPE's 8-bit codons address at most 256
productions, so the one dataset whose encoded width exceeds that (Ames
housing, 317 columns) uses 9-bit codons for every one of its cells, with
the search budget unchanged.

The cached LLM prior (`geprior/priors.py`, provenance recorded there,
including the basis of each pick) is one response per dataset and
therefore a fixed subset at these budgets; the priors for the thirteen
added datasets were written from the feature names and descriptions
before any selector or learner was run on them. A live per-draw prior
needs `LLM_BACKEND = "api"` and an `ANTHROPIC_API_KEY`. No ANOVA-F method
is included: on a binary label F is monotone in |r|. The NHANES label is a
threshold on RIDAGEYR, which the FeatLLM paper keeps as a feature, so any
selector that keeps it saturates the task; it is retained because the
paper retains it.

## Datasets

Eleven FeatLLM datasets under `data/featllm/` (budgets as in the earlier
campaigns) and thirteen benchmark datasets under `data/benchmark/`
(budget K = max(2, round(0.3337 d)), Memar et al.'s one-third rule).
`data/benchmark/PROVENANCE.md` gives every file's origin, checksum and
preparation; `fetch_benchmark.py` rebuilds the folder from the public
mirrors. The five families the two benchmarks share (adult, bank, heart,
german credit, communities) are kept in their FeatLLM form and not
duplicated.

| Collection | Classification | Regression |
|---|---|---|
| FeatLLM (11) | adult, bank, blood, car, communities, credit-g, diabetes, heart, myocardial, cultivars, NHANES | - |
| Benchmark (13) | breast cancer, dermatology, ionosphere*, spambase, optical digits*, mice protein, student dropout, sonar* | wine quality, geographical music*, parkinsons telemonitoring, student performance, ames housing |

\* anonymous feature names. `python -c "from geprior import tabular; print(tabular.describe().to_string())"`
prints the full table with roles, budgets and shot ladders.

## Analysis plan

Fixed before the thirteen added datasets were run. The campaign is
analysed in families, each on its own metric and its own blocks, written
to `results/<design>/analysis/<family>/`:

| Family | Datasets | Role |
|---|---|---|
| `classification` | the 19 classification datasets | primary |
| `regression` | the 5 regression datasets | descriptive (see below) |
| `classification_named` | the 15 classification datasets with named features | moderator |
| `classification_anonymous` | the 4 with anonymous features | descriptive |

Within a family. Confirmatory: at each shot level, the LLM prior against
each of correlation, mutual information, PCA loadings and RF importance,
blocked on dataset (per-dataset mean over draws), two-sided Wilcoxon
signed-rank, Holm across the four contrasts within the level, Cliff's
delta as the effect size. Semantic floor: the LLM prior against the random
subset at each level, Holm across the five few-shot levels (the full level
carries its raw p). Omnibus: Friedman over the five selection methods at
the five few-shot levels, Nemenyi post hoc with the critical difference
for that level's N; the all-methods omnibus over all seven methods at all
six levels is reported alongside. Exploratory, and labelled as such: an
all-pairs Wilcoxon with Holm across every pair within a level, per-dataset
Wilcoxon over the ten draws (Holm across levels), selection stability
(mean pairwise Jaccard across draws), subset overlap with the LLM prior.

Across families: a Friedman test on within-dataset ranks over all 24
datasets (ranks are metric-free; exploratory), the compute-time analysis
over every dataset (per-cell seconds paired within the (dataset, shot,
draw) cell against the all-features run), the contact table with Memar et
al.'s Table I, and the counts of regression cells at the R² floor.

Pre-specified moderator hypothesis. If the prior's standing against the
data-driven selectors rests on feature semantics, its contrasts should
favour it more in the named stratum than in the anonymous one, and the
semantic-floor contrast should separate the prior from the random subset
in the named stratum only. The anonymous stratum has four datasets, so it
is read descriptively.

Power. With N blocks a two-sided exact Wilcoxon cannot fall below 2/2^N:
with the 19 classification datasets the four-contrast Holm family can
reach alpha; with the 5 regression datasets it cannot (minimum
Holm-adjusted p = 0.125), so the regression family is pre-specified as
descriptive, reported for the direction and size of the differences and
for its Friedman omnibus. Medians are the primary summary; null results
are reported as findings. Where the Nemenyi post hoc disagrees with a
confirmatory contrast, both readings are reported side by side and the
confirmatory contrast is the pre-specified test.

### Record of changes to the plan

Each change is dated; none alters a confirmatory test retroactively.

1. Selector information budget (9 October 2026). The original plan, and
   the bundled reference run in `results/reference_run/`, used a matched
   design in which the data-driven selectors saw only the k labelled rows.
   After that run had been observed the selectors were given the whole
   training partition, which strengthens them into near-oracle baselines.
   The two designs are run as separate campaigns and never mixed.
2. Primary omnibus (6 October 2026). The omnibus reported as primary is
   the Friedman test over the five selection methods at the five few-shot
   levels; the pre-registered omnibus over all reported methods at all
   levels is retained alongside (`friedman_nemenyi_all.csv`).
3. Random-subset baseline (19 October and 28 October 2026). Pre-registered
   as a chance floor, excluded on 19 October as not answering a question of
   the full-information design, and reinstated on 28 October under a new
   reading: for a prior that never sees data, anonymising the names leaves
   no information about the columns, so the random subset at the same
   cardinality is the exact reference for the prior's semantic
   contribution, the data-free analogue of Memar et al.'s anonymised
   condition at ΔRetention = 0. The semantic-floor contrast is
   pre-specified for the extended campaign. Its random cells on the eleven
   FeatLLM datasets had been evolved before this plan was written (they
   sit in the bundled results), so on those datasets the contrast is not
   blind; on the thirteen added datasets it is.
4. Additional exploratory analyses (19 October 2026). The all-pairs
   Wilcoxon and the compute-time analysis were added after the first data
   had been seen and are reported as exploratory.
5. Extension to 24 datasets, task families and strata (28 October 2026).
   The thirteen datasets completing the Memar et al. benchmark, the
   regression task type with its fitness and floor, the family structure,
   the semantic-anchor moderator and the cross-task rank test were all
   specified before any of the added datasets was run. The eleven FeatLLM
   datasets' cells are the bundled ones from the 10 October (full
   information) and 2 October (matched) campaigns and were not re-evolved.
   The confirmatory family on the 19 classification datasets therefore
   contains eleven blocks whose values were known when the plan was
   extended and eight that were not.
6. Regression metric (28 October 2026, after the first run of the thirteen
   added datasets). The regression family had been specified with fitness
   and metric 1 − R² after a linear rescaling fitted on the k rows. On the
   completed run that metric was degenerate at the few-shot levels: the
   median test R² was at the −1 floor on four of the five datasets at
   k ≤ 16, with 60–99% of cells floored (wine quality 99% at k = 4), while
   the training R² of the same rules was ordinary. The cause is the
   calibration, not the search: a slope and intercept fitted on four to
   sixteen points, applied to protected divisions and logs whose test
   values fall far outside the training range, produce arbitrarily bad
   predictions; clipping the test values to the training range leaves
   wine quality floored in 81–96% of cells at k ≤ 16. Contrasts between
   methods at those levels would have compared ties. The regression
   fitness and metric were therefore changed to the Spearman ρ of the
   rule's output, the rank-based analogue of the AUC used throughout the
   classification family, and the five regression datasets were
   re-evolved under it; the calibrated R² (now with the clipping) is kept
   as the secondary metric and still used for the contact table with
   Memar et al. The change was made after observing the R² results of all
   seven methods, which were equally floored; it affects every method
   identically and no method's standing under R² was consulted in making
   it. The classification cells were not touched. The original R²-scored
   regression cells are kept in
   `results/fewshot_user_run_with_r2_regression.csv` for the record.

## Search-budget sensitivity analysis

The main run uses one GE budget (population 200, 40 generations) on every
dataset. Whether that budget should scale with dataset difficulty was
examined on the FeatLLM eleven as a sensitivity analysis: datasets tiered
by a convergence diagnostic stated before the re-run (hard = mean drop of
the best-so-far fitness over generations 30-40 at k = 64 of at least
0.005), the hard tier re-run with 80 generations and everything else
unchanged, only levels 32, 64 and full. `SEARCH_TIER` and `GE_BY_TIER` in
`geprior/config.py` record the tiers; datasets outside the table run at
the main budget. `Budget_Sensitivity.ipynb` and `Fitness_Convergence.ipynb`
drive this analysis; its results are never merged into the campaign file.

## Repository layout

    geprior/                 data loading, selectors, LLM prior, GE engine, runner
    grape_lib/               vendored GRAPE (de Lima et al.)
    grammars/                generated BNF grammars, one per (dataset, method)
    data/featllm/            the eleven FeatLLM datasets with metadata
    data/benchmark/          the thirteen benchmark datasets, metadata, raw files, PROVENANCE.md
    results/fewshot.csv      full-information campaign, one row per (dataset, shot, method, draw)
    results/matched/         matched campaign (seeded from the reference run)
    results/reference_run/   bundled matched-design run of 2 October 2026 (FeatLLM eleven)
    results/convergence/     harvested per-generation fitness at k = 64 (FeatLLM eleven)
    results/sensitivity/     search-budget sensitivity run and its analysis
    results/<design>/analysis/<family>/   tables and tests per family; report.md one level up
    figures/<design>/<family>/            figures per family; cross-family figures one level up

## Setup

Python 3.10 or later. From the project folder:

    ./setup.sh                      # Linux / macOS / WSL
    .\setup.ps1                     # Windows PowerShell

or by hand: create a virtual environment and `pip install -r
requirements.txt`. The `.vscode` folder carries launch configurations for
each step; open the folder in VS Code, select `.venv` as the interpreter
and the notebook kernel.

## Running

`Selector_Comparison.ipynb` is the main file: it drives either design
(`DESIGN` in its first cell), runs the campaign, the checks, the tables,
the report and the figures, and displays the headline tables and figures
inline. The same steps exist as commands:

    python run_fewshot_parallel.py            # full-information campaign; all cores but one; resumable
    python run_fewshot_parallel.py --matched  # matched campaign
    python run_study.py status                # progress per dataset
    python run_study.py share-invariant       # copy none/llm/random cells from the other campaign
    python check_reference.py                 # compare with the bundled reference and the other campaign
    python analyse_selectors.py               # tables and tests -> results/<design>/analysis/
    python report_selectors.py                # written report -> results/<design>/analysis/report.md
    python selector_figures.py                # figures -> figures/<design>/
    python standalone_figures.py              # per-method sheets and selector diagnostics
    python time_selectors.py                  # selector fitting cost
    python matched_campaign.py                # the matched design end to end from the command line
    python compare_designs.py                 # both designs' tables side by side -> results/comparison/
    python design_diagram.py                  # re-render design_diagram.png
    python fetch_benchmark.py                 # rebuild data/benchmark/ from the mirrors

Prefix any command with `GEPRIOR_SELECTOR_ROWS=shots` (or set it in the
environment) to address the matched campaign. A quick end-to-end check
before committing to the full run, about five minutes, written under
`results/_smoke`:

    python run_fewshot_parallel.py 2 --smoke

Compute. On a fresh checkout the eleven FeatLLM datasets are complete in
both campaigns (bundled), so what runs is the thirteen added datasets:
about 13 000 single-rule GE searches per campaign (the multiclass
datasets evolve one rule per class), roughly 14 CPU-hours at 4 s per
search; an afternoon on eight processes, an overnight run on four. The
design-invariant methods (all features, LLM prior, random subset) are
shared between the campaigns by `share-invariant`, so the second campaign
costs about four sevenths of the first. Interrupting is safe: completed
cells are skipped on re-run.

## Comparing the designs

`compare_designs.py` reads the analysis folders of both campaigns and
writes, per family, the pooled medians, the confirmatory and semantic-floor
contrasts, the Friedman mean ranks and the subset stability with one column
block per design, under `results/comparison/`, plus `comparison.md`. It
recomputes nothing; run both designs' analyses first.

## Figures

`selector_figures.py` draws, per family, pooled test score and mean rank
against k, critical-difference diagrams, paired per-dataset differences of
the LLM prior against each selector and against the random subset, the
confirmatory and all-pairs significance maps, stability and overlap; and,
across families, the semantic-anchor figure, the cross-task rank figure,
the compute-time figures and one score-against-k figure per dataset under
`figures/<design>/per_dataset/`. Figures carry no titles; the captions in
`figures/CAPTIONS.md` describe each one. `standalone_figures.py` draws each
method on its own (one result sheet per method and family) and the
selector diagnostics per dataset under `figures/<design>/selector_diagnostics/<dataset>/`.
