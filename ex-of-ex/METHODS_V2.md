# Protocol v2: evidence for incremental hidden information

The hypothesis is that final-layer hidden-token information can correct some
global retrieval errors. The experiment must also be capable of rejecting that
hypothesis for the tested readouts. Better protocol does not imply better numbers.
The encoder, checkpoint, token extraction and existing parameter-free scorers stay
frozen. Trained heads below operate on scorer outputs, not on raw hidden tokens;
a negative result is therefore not an upper bound on all information in the tokens.

## Method provenance and the requested 80/20 constraint

We interpret 80/20 as a constraint on methodological novelty, not a measurable
percentage of code or a claim of exact paper reproduction. No new matching loss
or new token scorer is introduced. The substantive mechanisms are established:

| Component | Published basis | Implementation / adaptation |
|---|---|---|
| Pairwise learned score | Burges et al., ICML 2005, [Learning to Rank using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/) | Linear RankNet logistic pair loss, L2 regularization; input is frozen global/local scalar scores. Mirrored pairs and no intercept enforce swap antisymmetry. |
| Nonlinear global control | Zadrozny & Elkan, KDD 2002, [Transforming Classifier Scores into Accurate Multiclass Probability Estimates](https://www.cs.columbia.edu/~djhsu/coms4771-f25/handouts/zadrozny2002kdd.pdf) | Isotonic calibration of the global pair margin; separates nonlinear global calibration from hidden-score gain. |
| Selective intervention | Geifman & El-Yaniv, NeurIPS 2017, [Selective Classification for Deep Neural Networks](https://papers.nips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) | Global-margin confidence and a logistic correctness predictor for the proposed reranker; identity-disjoint fitting/selection. |
| Independent risk check | Angelopoulos et al., [Learn then Test](https://arxiv.org/abs/2110.01052) | Choose one gate per budget on selection, test on calibration, Bonferroni across budgets, revert to no-op on failure. LTT-inspired adaptation, not a reproduction of its full algorithm. |
| Bootstrap | Efron, 1979, [Bootstrap Methods: Another Look at the Jackknife](https://blogs.helsinki.fi/bk-club/files/2012/05/Efron_Bootstrap_AS1979.pdf) | 2,000 paired identity-cluster draws; percentile 95% intervals. Identity aggregation is the TBPS adaptation. |

Task-specific engineering is limited to score features, identity grouping,
candidate bookkeeping, harm definition and CSV/NPZ reporting. Existing scorers
retain their earlier paper-inspired status: these functions are not reproductions
of complete SCAN/CFine/etc. models and training procedures.

## Four experiments and fixed decisions

1. **Setwise diagnostic**: all gallery positives plus the first M global hard
   negatives, M=1,5,10. Compare parameter-free scorers on identical candidates.
   `setwise_summary.csv` includes accuracy, recovery/harm, paired differences from
   MaxSim, and CIs. `setwise_learned_summary.csv` adds the validation-selected
   RankNet probe head, its global-only reference, and the matched hidden-only head.
   Keep trained and parameter-free evidence separate. These are
   oracle candidate sets, not deployable retrieval. M=1 with all positives is not
   the original single-P versus W1 diagnostic.
2. **Controlled probe**: fit linear pair heads on FIT IDs; choose C in [0.1,1,10]
   using SELECTION weighted log-loss; evaluate on TEST. Test contains all positive
   versus first 10 hard-negative pairs, weighted equally per person. Compare
   global-only, global isotonic, each hidden-only, global+one hidden, global+all
   hidden, and matched shuffled controls. Hidden columns are shuffled jointly
   AFTER orientation, independently in each split. No label-derived sign is
   applied afterward. The primary combined head and global reference are selected
   before looking at test. AUC CIs are computed for those primary comparisons and
   matched control; blank AUC CI for exploratory rows means not computed. All
   probe rows have accuracy/log-loss CIs and paired log-loss differences.
3. **Actual retrieval**: fixed global top-K, K=10,20,50, without positive injection.
   Candidates include earlier fusion/local variants and learned linear RankNet
   heads. Choose by selection net rescued (=R1 improvement), with lower harm and
   no-op preferred on ties. TEST is evaluated only for preselected overall,
   global baseline, and MaxSim-or-no-op reference. Per-query exact R1/R5/R10/AP/INP
   and paired CIs are saved. Uniform control and learned global-only can win;
   such a win is not evidence for fine-grained hidden matching.
4. **Selective gating**: take the best non-noop retrieval candidate on selection.
   Fit a logistic correctness model on FIT disagreements using global margin,
   reranker margin, local advantage and global cost. Compare its confidence with
   negative global margin. Select threshold/source per budget on SELECTION by
   positive net rescued under empirical micro harm <= budget. Reject negative
   net gains. Output two explicitly different policies: empirical (no calibration
   guarantee) and risk_screened (independent calibration check). An empty or
   rejected proposal uses no-op. Calibration never chooses a new candidate or
   searches for another threshold after rejection.

## Risk meaning: macro and micro are different

Report the original micro harm = harmed queries / globally correct queries.
For the independent check, each calibration identity with at least one globally
correct query is one observation. Its Bernoulli loss is 1 if ANY previously
correct query is harmed. A one-sided Clopper-Pearson upper bound is computed at
delta/number_of_budgets. That conservative loss upper-bounds identity-macro
conditional harm (equal weight per eligible person). It does NOT certify the
query-micro harm ratio. Both are labeled in output. This is a deliberate
conservative cluster adaptation of split risk testing, not a distribution-free
guarantee for a fixed benchmark or for distribution shift. It assumes independent,
exchangeable identity clusters conditional on the fitted system and gallery.

With 50 eligible calibration IDs, zero harmed IDs still gives an upper bound
about 7.86% at delta=0.05/3. Thus 1%,2%,5% budgets may all fall back to no-op on
RSTP. This means insufficient calibration evidence, not zero hidden information.
The empirical ablation remains visible. More bootstrap draws cannot overcome a
small calibration sample. Never use a zero bootstrap percentile from zero observed
harm as a finite-sample risk certificate.

## Splits and ICFG

For CUHK/RSTP, official validation QUERY IDs are partitioned once with a fixed
seed into 50% FIT, 25% SELECTION, 25% CALIBRATION. Official test stays the evaluation
set. Each partition keeps its original gallery, so gallery distractor difficulty
does not shrink. Query IDs are disjoint; galleries are shared within a source.
This is conditional-on-gallery evidence, not a fully independent-gallery test.

The upstream `diagnostic` runner has training=False and uses dataset.test. Its
`val_dataset='test'` default does not create an official validation split.
See [upstream loader](https://github.com/HoangVo-Prog/probably-works/blob/diagnostic/datasets/build.py)
and [upstream runner](https://github.com/HoangVo-Prog/probably-works/blob/diagnostic/diagnostic/global_hidden_recovery/run_diagnostic.py).
The previous ex-of-ex validated runner incorrectly required val unconditionally.

ICFG v2 explicitly opts into test_identity_holdout: 20% of official test QUERY IDs
become development, partitioned as above; 80% remain evaluation. The official test
gallery is fixed. Record every ID in split_assignments.csv. Do not present this
as an official full-test result or compare its R1 directly with the old 66.04%.
Compare against the v2 global baseline on the SAME held-out queries. The upstream
diagnostic command and top-level full-test baseline remain available unchanged.
No encoder-training identities are relabeled as unseen validation identities.

## Bootstrap and interpretation

2,000 samples with replacement of identity clusters, preserving all observations
of each selected identity. Compare methods using the SAME draw. R1/mAP changes
are percentage points; recovery/harm and setwise/probe accuracy are fractions;
AUC is unitless. Undefined denominators produce NaN/JSON null. If fewer than 90%
of draws are valid, the interval is marked undefined. This concerns sparse data,
not model failure. Fix the seed and repetitions in advance.

Percentile CIs are descriptive and conditional on the chosen head, split,
checkpoint and gallery. They do not account for re-training/selection variability,
are not simultaneous CIs for all exploratory rows, and do not constitute a new
held-out confirmation after v1 test results have already been inspected. Primary
comparisons are chosen on selection. A follow-up confirmed claim needs a fresh
benchmark/holdout; do not pick a new scorer or seed by maximum test recovery.

## Outputs and reproducibility

New runs use validated_v2/ inside the timestamped run directory. Existing v1
outputs are not rewritten. The shared run.log covers all four experiments;
protocol.json and summary.json give configuration/scope. CSVs store metrics and
CI columns, models JSON stores fitted parameters, NPZs store predictions and
per-query arrays for recomputing uncertainty without CLIP. AUC CIs are exact
weighted paired bootstrap for the preselected comparison, not a normal approximation.

Run `python ex-of-ex/tests.py` and `python ex-of-ex/tests_v2.py` for CPU regressions.
The full dataset experiments still require Kaggle/GPU (or substantially longer CPU
scoring). See KAGGLE_CELLS.md and run_cached_v2.py for cached-feature execution.
