# KPI-RAG: Explainable Root-Cause Analysis for 5G Networks
## Phase 2 Final Report

**Project:** KPI-RAG
**Track:** AI & Data Science
**Program:** DiGiLiANS Master Program — Round 1
**Supervisor:** Prof. Yuanzhu Chen
**Industry Partner:** Capgemini
**Institution:** National Telecommunication Institute (NTI)
**Team:**
- **Ahmed Mohamed Al Shobaki** — Team Leader, RAG Pipeline, LLM & Dashboard
- **Farida El Ghabary** — Data Pipeline & Feature Engineering
- **Raneem Alnaghy** — ML Models & SHAP Attribution
- **Rodina Khallaf** — Standards Grounding & Evaluation

**Submission Date:** October 2026
**Live Demo:** https://kpi-rag-group15.streamlit.app
**Repository:** https://github.com/KPI-RAG/kpi-rag

> **Sources.** Every number below is read from a file in the repository; Appendix F lists them.
> Results that do not exist yet (Track B G-Eval on these explanations) are reported as pending, not estimated.

---

## Abstract
5G Radio Access Networks expose dozens of KPIs per cell, and diagnosing a fault means relating an
anomalous KPI pattern to its physical cause and to the 3GPP specification that governs it. KPI-RAG
is an end-to-end pipeline that does this explainably. On the TelecomTS dataset (32,000 windows of 18
KPI channels, 128 samples at 10 Hz), a two-stage CatBoost cascade detects faults (binary recall
**88.7 %**, 266/300; precision 69.6 %) and classifies them into 11 types (exact accuracy **66.3 %**
on all test faults; 51.5 % when the single real-world class, Jamming, is excluded). TreeSHAP
attributions and the window's statistics drive retrieval of past incidents of the predicted fault
type from a train-only ChromaDB index, and an LLM (Gemini 3.5 Flash Lite) writes a structured
explanation grounded in a fault-to-3GPP alignment table and per-window evidence. In a 30-window
ablation sampled by ground truth and run through the full system, the cited standard is correct for
the **true** fault in 5/30 explanations with label and SHAP only (C1), 7/30 with retrieved incidents
(C2) and **21/30 with the alignment table and evidence (C3)** — a gain of 47 percentage points over
ticket-only context. When the fault is classified correctly, so that the correct standard is in the
prompt, the LLM cites it in **19/19** cases. C3 matches a rule that cites the predicted fault's table
row exactly (21/30), which confirms the alignment table as the grounding mechanism: the LLM reliably
adopts the provided standard. All nine C3 misses are detector misses or misclassifications upstream.
Explanation-quality scoring (G-Eval) is pending.

## Acknowledgements

We thank our supervisor, **Prof. Yuanzhu Chen**, for his guidance throughout the project; our
industry partner **Capgemini** for operational perspective on NOC troubleshooting; the
**National Telecommunication Institute (NTI)** and the **Ministry of Communications and
Information Technology (MCIT)** of Egypt for the **DiGiLiANS Master Program**; and the authors
of the **TelecomTS** dataset [13].

---

## Table of Contents

1. Introduction — 1.1 Background · 1.2 Problem Statement · 1.3 Objectives · 1.4 Research Questions · 1.5 Report Organization
2. Background — 2.1 5G and O-RAN · 2.2 KPIs in TelecomTS · 2.3 Fault Types · 2.4 RAG · 2.5 SHAP · 2.6 G-Eval · 2.7 Gap
3. Methodology — 3.1 Dataset · 3.2 Fault Distribution · 3.3 Features · 3.4 Cascade · 3.5 SHAP · 3.6 Retrieval · 3.7 Alignment Table · 3.8 RCA Evidence · 3.9 LLM · 3.10 Evaluation Design
4. Design and Implementation — 4.1 Architecture · 4.2 Data Pipeline · 4.3 Layer 2 · 4.4 Retrieval · 4.5 Explanation · 4.6 RCA Loader · 4.7 Dashboard · 4.8 Engineering Decisions
5. Results — 5.1 Track A Detection · 5.2 Track A Classification · 5.3 Track C · 5.4 Track B (G-Eval) · 5.5 RAG Retrieval · 5.6 Tests
6. Discussion — 6.1 Interpretation · 6.2 G-Eval Analysis · 6.3 Key Technical Findings · 6.4 Objectives Met · 6.5 Limitations · 6.6 Practical Implications
7. Conclusion and Future Work
References · Appendices A–F

---

## 1. Introduction

### 1.1 Background
5G NR and Open RAN disaggregate the base station into radio, distributed and central units with
open interfaces, which multiplies the number of components and KPIs an operations team must
watch. When KPIs degrade, the time to identify the root cause and the applicable standard drives
the mean time to repair.

### 1.2 Problem Statement
An engineer confronted with an anomalous window must (1) recognise which fault the KPI pattern
indicates, (2) explain why, and (3) find the governing 3GPP clause among hundreds of
specifications. Classifiers answer (1) without explanation. LLMs can produce fluent explanations
but, without grounding, cite specifications that do not govern the fault (Section 5.3: 5/30
correct in C1).

### 1.3 Objectives
1. Detect and classify 11 fault types with a leakage-free, session-split cascade.
2. Attribute each prediction to the KPIs that drove it (SHAP).
3. Retrieve past incidents comparable to the current window.
4. Ground the explanation in a curated fault-to-3GPP alignment table and verify the citation.
5. Quantify the contribution of each context source with a controlled ablation.

### 1.4 Research Questions
- **RQ1:** Do SHAP attributions and retrieved incidents improve citation correctness over an
  LLM given only the predicted label?
- **RQ2:** Does injecting the fault's 3GPP alignment row make the cited standard correct, and how much
  of the remaining error comes from the classifier rather than the grounding?
- **RQ3:** Are the explanations specific, actionable and causally sound (G-Eval)?

### 1.5 Report Organization
Chapter 2 covers background, Chapter 3 the method, Chapter 4 the implementation, Chapter 5 the
results, Chapter 6 the discussion and limitations, Chapter 7 conclusions.

---

## 2. Background and Literature Review

### 2.1 5G Network Architecture and O-RAN
The O-RAN Alliance [12] splits the gNB into the **O-RU** (RF front end, low PHY), the **O-DU**
(high PHY, MAC, RLC, schedulers, buffers) and the **O-CU** (PDCP, SDAP, RRC), with the
**Near-RT RIC** hosting xApps for control loops between 10 ms and 1 s (mobility, traffic
steering, resource allocation). Localising a fault to one of these components is part of a
useful diagnosis.

### 2.2 Key Performance Indicators in 5G (TelecomTS)
Each TelecomTS window carries 18 channels [13]: 16 numeric KPIs — `RSRP`, `DL_BLER`, `UL_BLER`,
`DL_MCS`, `UL_MCS`, `UL_NPRB`, `UL_SNR`, `TX_Bytes`, `RX_Bytes`, `Estimated_UL_Buffer`,
`PRBs_DL_Current`, `PRBs_UL_Current`, `PRB_Utilization_DL`, `PRB_Utilization_UL`,
`UL_NumberOfPackets`, `DL_NumberOfPackets` — and two categorical protocol fields (`UL_Protocol`,
`DL_Protocol`). Windows also carry a natural-language KPI narrative (`description`) and a Q&A trace.

### 2.3 Fault Types in TelecomTS
Ten synthetic fault types grounded in telecom literature plus one real over-the-air anomaly
(Jamming) [13]: Antenna Failure; Buffer Overflow (Gradual Buildup); Co-Channel Interference
(Mild, Severe); Doppler Shift (Severe); Faulty Handover Algorithm (Too Frequent); Faulty RF
Filters (Temporal); High Network Congestion (Gradual Buildup, Sudden Spike); Resource Allocation
Bugs; Jamming.

### 2.4 Retrieval-Augmented Generation
RAG [1] conditions a language model on passages retrieved from an external store, typically with
a bi-encoder such as Sentence-BERT [5] and approximate nearest-neighbour search. Its value depends
on the corpus containing information the model lacks — a point that turns out to matter here
(Section 6.3).

### 2.5 SHAP for Model Explainability
SHAP [2] assigns each feature an additive Shapley contribution to a prediction; TreeSHAP computes
it exactly for tree ensembles such as CatBoost [4]. The sign of a SHAP value says whether the
feature pushed toward the predicted class, not whether the KPI was high or low; the project
therefore reports direction separately (`feature_vs_normal`, Section 3.5).

### 2.6 G-Eval Framework
G-Eval [3] scores generated text with an LLM judge that follows explicit rubric dimensions on
Likert scales. Track B applies it (Rodina Khallaf) to the Track C explanations on the four
dimensions in Appendix C. A judge from the same model family as the generator can favour its own
outputs, so a human spot-check of a subset is planned.

### 2.7 Gap Analysis
Work on automated RAN fault management concentrates on classification accuracy; LLM assistants
for telecom are rarely evaluated on whether their standards citations are correct for the fault.
KPI-RAG combines attribution, retrieval and a verifiable citation check, and measures the
citation end to end against ground truth.

---

## 3. Methodology

### 3.1 Dataset: TelecomTS
| Item | Value | Source |
|---|---|---|
| Windows | 32,000 (1,235 anomalous, 30,765 normal) | `data/raw`, `build_index.py` log |
| Window | 128 samples at 10 Hz (12.8 s), 18 channels | raw records (`sampling_rate: 10`) |
| Session split — train | 25,688 windows (935 anomalous) | `train_idx_sessionsplit.npy` |
| Session split — test | 6,312 windows (300 anomalous, 6,012 normal) | `test_idx_sessionsplit.npy`, `validation.json` |

Windows are split by session so that no test window shares a recording session with training data.

### 3.2 Fault Type Distribution (test split)
| Fault type | True test windows | Phase 2 predictions | Correctly classified |
|---|---:|---:|---:|
| Jamming | 94 | 93 | 93 |
| High Network Congestion (Gradual Buildup) | 37 | 2 | 2 |
| Faulty RF Filters (Temporal) | 33 | 55 | 25 |
| Co-Channel Interference (Severe) | 30 | 12 | 12 |
| Buffer Overflow (Gradual Buildup) | 21 | 46 | 15 |
| Co-Channel Interference (Mild) | 20 | 88 | 9 |
| Faulty Handover Algorithm (Too Frequent) | 19 | 30 | 10 |
| Resource Allocation Bugs | 17 | 17 | 17 |
| High Network Congestion (Sudden Spike) | 11 | 1 | 0 |
| Antenna Failure | 9 | 21 | 7 |
| Doppler Shift (Severe) | 9 | 17 | 9 |
| **Total** | **300** | **382** | **199** |

Predictions include the 116 false alarms on normal windows (64.7 % of them labelled CCI Mild,
`validation.json`). Sources: `layer2_output_sessionsplit_phase2.json`, `layer2_rag_handoff_sessionsplit.json`.

### 3.3 Feature Engineering (Phase 2, Farida El Ghabary)
- **Model 1 (64 features):** mean, std, min, max of the 16 numeric KPIs.
- **Model 2 (496 features,** `feature_spec.json`, recipe "Farida v3 500/class + Config B + FFT + G1G2G3"):
  64 base statistics; 256 temporal-patch statistics (8 patches × 16 KPIs × mean/std); 80 spectral
  features (4 FFT band energies + total energy × 16 KPIs); 84 temporal-shape features (20 shift,
  16 log-std-ratio, 16 trend, 16 convexity, 16 S-shape); 2 onset; 6 cross-KPI correlations
  (e.g. `corr_UL_SNR_UL_MCS`, `corr_DL_MCS_DL_BLER`); 3 difference features; 1 radio-share feature.

### 3.4 Two-Stage CatBoost Cascade
| Stage | Model | Features | Trees | Training rows | Source |
|---|---|---:|---:|---:|---|
| 1 — detect | binary CatBoost, threshold 0.5 | 64 | 200 | 4,675 (80/20 rebalanced) | `model1.cbm`, `log.txt` |
| 2 — classify | 11-class CatBoost | 496 | 1,000 | 5,185 (real training faults + synthetic windows) | `model2.cbm`, `log.txt` |

Stage 2 runs only on windows Stage 1 flags. An isotonic calibrator for Stage 1 was evaluated and
rejected (`log.txt`: "calibration rejected in step 12a"). The `confidence` of an alarm is Stage 2's
uncalibrated probability for the predicted fault type (62 of the 382 alarms are below 0.5); for a
normal window it is Stage 1's probability of Normal (`phase2/src/pipeline.py`).

**Model selection.** The split is by session and no session appears in both parts. Layer 2 models
were selected by the ML team (Farida El Ghabary, Raneem Alnaghy) based on internal validation
criteria; because the same test sessions were also consulted during model development, the Track A
figures should be read as a test-split estimate rather than a fully untouched one. The synthetic
Model 2 windows are built from training sessions only.

### 3.5 SHAP Attribution
TreeSHAP values are computed for the predicted class; the top three features with a single KPI
channel are kept. Layer 2 stores each value as a magnitude (|SHAP|) with its sign as `shap_effect`
(`toward_predicted_class` / `away_from_predicted_class`; 97 of the 1,146 top-3 entries point away),
plus whether the window's feature value lies above or below the normal-traffic mean
(`feature_vs_normal`, from `normal_means.json`). Prompts show all three — e.g. "RSRP: below normal
(|SHAP|=0.42, toward predicted class)" — so neither the KPI direction nor the effect is read from a sign.

### 3.6 RAG Pipeline
- **Corpus:** the 935 anomalous training windows only; one document per window — its KPI
  narrative (`description`). Indexed with `all-MiniLM-L6-v2` in ChromaDB (cosine HNSW).
- **Query:** the window's KPI evidence (top SHAP channels, their directions, UL/DL means). The
  fault label is **not** in the query text.
- **Filter:** results are restricted to incidents of the **predicted** fault type
  (`where={"anomaly_type": …}`); top 5 are returned, top 3 shown to the LLM.
- **Why this design** is a measured result, not a preference (Section 6.3).

### 3.7 3GPP Alignment Table v1.2 (Rodina Khallaf)
Eleven rows map each fault type to Release 19 clauses and an O-RAN component (Appendix A).
Ten rows are `confirmed`; Jamming is `not-applicable` (an external over-the-air event with no
conformance clause). A citation is valid for a fault if it is a well-formed TS/TR number listed in
that fault's row; a fault without a row (Jamming) can never have a valid citation.

### 3.8 RCA Evidence Integration
`rca_evidence.json` holds 289 records — one per non-Jamming Phase 2 alarm, all test windows, covering
10 fault types. For C3, a compact block (537–659 characters) is injected when the record is for the
same window and predicted fault: the top-3 SHAP features (magnitude, direction, effect), the window's
observed means of the KPIs behind them, and the fault's causal mechanism. Two parts of the record are
deliberately not injected: its own standards text (167 records differ from table v1.2, e.g. Handover
still cites TS 38.133 — the alignment table is the only source of standards in the prompt) and its
KPI "evidence status" labels, which derive from class-level expected KPIs aggregated over labelled
windows that include the test split. The record never contains the window's true label.

### 3.9 LLM Backend
Gemini 3.5 Flash Lite, temperature 0.1, native JSON output, ≥ 2 s between calls (thread-safe across
dashboard sessions). Rate-limit (429) errors wait for the server's `Retry-After` up to 60 s (else
exponential backoff, 5–60 s); a longer requested wait (e.g. a daily quota) ends the attempt instead of
blocking. Client errors (4xx other than 408/409/429, e.g. an invalid key or a removed model) and
configuration errors stop immediately; malformed replies are retried; every failed attempt is logged;
after three attempts a template explanation is returned, counted as invalid and labelled with its
cause. Citations are normalised to `TS/TR XX.XXX`. Groq and Ollama are supported alternatives.

### 3.10 Evaluation Framework
- **Track A:** detection and classification metrics recomputed from the Phase 2 predictions and
  ground truth (`scripts/run_eval_track_a.py`), plus retrieval diagnostics
  (`scripts/experiment_retrieval_design.py`).
- **Track C (citation ablation):** 3 test windows per **true** fault type (10 types; Jamming has no
  clause), `random_state = 42`. Each window goes through the full system: its Phase 2 prediction,
  SHAP and RCA record, retrieval filtered by the predicted type, and the LLM under three conditions
  that share one instruction block and differ only in context:
  C1 label + SHAP; C2 + three past incidents; C3 + alignment row + RCA evidence.
  Each citation is scored against the **true** fault's row (`citation_correct`, end to end) and
  against the predicted row (`reference_valid`, what the validator reports). A window the detector
  misses produces no explanation and counts as incorrect. Two rules that need no LLM are scored the
  same way: *always cite the single best standard*, and *copy the predicted fault's table row*.
- **Track B (G-Eval, LLM judge):** four 1–5 dimensions on every Track C explanation, judged against the true fault's alignment row (Appendix C).

## 4. Design and Implementation

### 4.1 System Architecture
```
LAYER 1  Data        TelecomTS JSONL ── session split (train / test indices)
LAYER 2  Detection   Model 1 (64 f) ── Model 2 (496 f) ── TreeSHAP          [offline, phase2/ on main]
                     └─► layer2_output_sessionsplit_phase2.json (6,312 test windows)
LAYER 3  Grounding   KPI-evidence query ── ChromaDB (train-only, filtered by predicted type)
                     alignment table v1.2 ── RCA evidence
LAYER 4  Reasoning   Gemini ── citation validator ── Streamlit dashboard
```

### 4.2 Data Pipeline
`src/data_loader.py` reads the JSONL files in sorted order, keeps anomalous records and flattens
them into tickets. Raw records carry no id, so a ticket's id is its position; `verify_ticket_order()`
checks every ticket's label against the ground truth for that index and `apply_train_split()`
raises if the split file is missing, so a reordered dataset or missing split fails loudly instead
of leaking test windows into the index.

### 4.3 ML Pipeline (Layer 2, offline)
Layer 2 is trained and run offline by Farida El Ghabary and Raneem Alnaghy with the pipeline in
`phase2/` on the team's `main` branch (`scripts/11_final_pipeline.py`), which needs the parsed-signal
caches on the project's JupyterHub server. This branch consumes its output JSON and model files;
`src/ml_model.py` loads the models but has no feature builder and is used only by tests. Both model
files store anonymous feature names (`'0','1',…`), so column order is defined by `feature_spec.json`.

### 4.4 RAG Pipeline
`src/kg_indexer.py` embeds each training window's description; `scripts/build_index.py --rebuild`
recreates the collection from the raw data (a plain run skips existing ids, so `--rebuild` is what
guarantees no stale documents). `src/rag_query.py` builds the label-free evidence query and applies
the predicted-type filter.

### 4.5 LLM Explanation Pipeline
`src/llm_explainer.py` builds the three prompts from shared pieces — header (label, confidence,
SHAP with magnitude, direction and effect), optional incidents block, optional RCA and standards
blocks, one common instruction block — calls the LLM, parses the JSON (bad JSON is retried),
normalises the reference to `TS/TR XX.XXX` and validates it. Citation validation is at specification
level; the clause number is not checked.

### 4.6 RCA Loader
`src/rca_loader.py` indexes `rca_evidence.json` by window and formats the C3 evidence block (Section 3.8).

### 4.7 Dashboard (Streamlit)
`dashboard/app.py` (Streamlit) has six panels: Detection (label, Stage 2 probability of that label),
SHAP (bar chart), KPI Window Statistics, Explanation (with a ✅/❌ citation badge and, for a template
fallback, its cause), Sources (retrieved incidents with similarity, flagged when weakly similar) and
RCA Evidence (standard and O-RAN component taken from the alignment table). "Browse" lists the 382
Phase 2 alarms (all test windows, so none has its own ticket in the index), filtered by fault type;
RCA evidence is injected only when the record matches the window's predicted fault. Uploaded payloads
are keyed by content, so a new upload never shows the previous file's explanation. The required API
key follows the configured backend, and all paths resolve from the repository root.

### 4.8 Key Technical Decisions
- **Cosine index.** An earlier index used L2 distance while code computed similarity as 1 − d; it
  was rebuilt with cosine space. `build_index.py --rebuild` refuses to clear the index when no raw
  tickets were loaded.
- **Calibrator removed** (Section 3.4).
- **Single sources:** one alignment table (`data/alignment_table.json`, read through the config) for
  every standard shown or injected; one dependency lock (`uv.lock`), from which `requirements.txt` is
  exported.
- **Fail-closed guards:** missing split files abort indexing and Track C instead of degrading silently;
  a Track C run does not overwrite an earlier one without `--force`.
- **Deployment:** CPU-only PyTorch (no GPU is used; the default Linux wheel pulled about 2 GB of CUDA
  libraries), Python ≥ 3.11, runtime and test dependencies separated, `src/` installed as a package so
  scripts run without `PYTHONPATH`. The split indices are committed so a clean clone runs the tests.

## 5. Results and Evaluation

### 5.1 Track A — Binary Detection (`data/processed/track_a_metrics.json`)
| Metric | Value |
|---|---|
| Test windows / true faults | 6,312 / 300 |
| Flagged by Stage 1 | 382 |
| True faults detected | 266 |
| False alarms | 116 |
| **Binary recall** | **88.7 %** (266/300) |
| **Binary precision** | **69.6 %** (266/382) |

### 5.2 Track A — 11-Class Classification
| Metric | Value |
|---|---|
| Exact 11-class accuracy, all true faults | **66.3 %** (199/300) |
| Exact accuracy, detected faults only | 74.8 % (199/266) |
| Accuracy with the congestion family merged (Buffer Overflow + both Congestion types) | 75.0 % (225/300) |
| Exact accuracy excluding Jamming | 51.5 % (106/206) |
| Binary recall excluding Jamming | 83.5 % (172/206) |

Jamming — the one real over-the-air anomaly — is 94 of the 300 test faults and 93 of the 199 correct
classifications, so the headline accuracy leans on it. The recomputed counts equal Farida's
`validation.json` (checked by `tests/test_run_eval_track_a.py`).

### 5.3 Track C — Citation Accuracy Ablation (`data/processed/track_c_v3/track_c_summary.json`)
Run v3 (`data/processed/track_c_v3/`): 30 windows (3 per true fault), 72 LLM calls, 0 template
fallbacks, alignment table v1.2. Run v2 (`track_c_v2/`, before the evidence-block fixes of Section 3.8)
gave identical counts on the same windows.

> **Methodology note.** Sampling is by **ground truth**, citations are scored against the true fault,
> and all three conditions share one instruction block. The first Track C run
> (`data/processed/track_c_explanations.jsonl`) sampled by predicted label (8 of its 30 windows were
> normal traffic, 9 misclassified) and C3 alone carried extra instructions; its figures (C1 8/30,
> C2 7/30, C3 30/30 against alignment v1.1; 28/30 against v1.2) are kept for comparison only.

| Answer rule | Correct vs true fault (n = 30) | Wilson 95 % CI | When classified correctly (n = 19) | Valid vs predicted row (24 explained) |
|---|---:|---|---:|---:|
| C1 LLM: label + SHAP | 5/30 (16.7 %) | 7.3–33.6 % | 4/19 | 7/24 |
| C2 LLM: + past incidents | 7/30 (23.3 %) | 11.8–40.9 % | 6/19 | 9/24 |
| C3 LLM: + alignment row + RCA evidence | **21/30 (70.0 %)** | 52.1–83.3 % | **19/19** | 24/24 |
| No LLM: always cite TS 38.141-1 (best single standard) | 12/30 (40.0 %) | — | — | — |
| No LLM: copy the predicted fault's row | 21/30 (70.0 %) | — | 19/19 | 24/24 |

- **Alignment-table grounding is effective.** When the correct standard is provided (the 19 correctly
  classified windows), the LLM incorporates it in 19/19; end to end it is 21/30 (70 %). The C2→C3 gain
  (+47 percentage points) quantifies the contribution of the alignment table over ticket-only context.
- **The copy-row baseline confirms the mechanism.** C3 equals the rule that cites the predicted fault's
  row (21/30): the alignment table is the critical grounding mechanism, and the LLM reliably adopts the
  provided standard (24/24 against the predicted row), which is the intended behaviour of the system.
  C3's end-to-end accuracy therefore rises with classifier accuracy.
- **Without the table, citations are unreliable.** C1 (5/30) and C2 (7/30) are below a fixed answer
  ("always cite TS 38.141-1", 12/30). C1 mostly answers TS 38.104 (11), TS 38.331 (8) and TS 38.214 (4),
  plus one LTE specification (TS 36.104). Retrieved incidents change little: C2 improves 2 windows over
  C1 and worsens none.
- **Where C3 fails (9/30):** 6 detector misses — all three Sudden Spike samples (866, 827, 843), one
  Congestion Gradual (682), one Buffer Overflow (717), one Handover (879) — and 3 misclassifications
  whose injected row is for the wrong fault (845, 846: Congestion predicted as Buffer Overflow; 648:
  Handover predicted as RF Filters). Two misclassified windows were still correct (645, 703) because
  the wrong row cites the same specification.
- **Statistics are descriptive.** C2→C3: 14 windows better, 0 worse (exact McNemar p = 1.2·10⁻⁴);
  C1→C3: 16 better, 0 worse. The windows come from a small number of recording sessions (Layer 2 has
  10 fault sessions in total) and are sampled 3 per class, so they are neither independent nor a
  population sample; the intervals and p-values indicate magnitude, not significance.

### 5.4 Track B — G-Eval Quality Assessment (pending)
Pending. Rodina Khallaf's G-Eval judge was first run on the earlier Track C explanations, which
were sampled by predicted label: of its three "Antenna Failure" windows one was CCI Severe and two
were normal traffic, and its three "Sudden Spike" windows were two CCI Severe and one normal window.
Those scores therefore describe a different, partly mislabelled sample and are not reported here.
The judge is being re-run on the 72 explanations in
`data/processed/track_c_v3/track_c_scores_template.jsonl` (each tagged with its true fault), using
the alignment table v1.2 row of the true fault as the citation reference. `src/evaluator.py` rejects scores outside 1–5, so an un-rated template cannot be
averaged by mistake. The planned aggregate is the equal-weight mean of the four dimensions per
explanation, and the pre-registered pass mark (`citation_validity_threshold = 0.70`) applies to
the rate of correct citations.

### 5.5 RAG Retrieval Quality
Measured on the 266 detected true faults with `scripts/experiment_retrieval_design.py` (in memory;
the deployed-design row is from `track_a_metrics.json`).

| Design | P@1 / P@5 vs true class |
|---|---|
| Previous: class-template documents, label in the query | 0.707 / 0.686 — below the classifier's 0.748 |
| Template documents, label removed from the query | 0.102 / 0.069 |
| Per-window KPI narratives, label-free query (deployed ranking) | 0.451 / 0.335 (majority-class share 0.353) |
| Same, averaging all paragraph embeddings to avoid truncation | 0.049 / 0.201 |
| Deployed design (filter by predicted type), top-1 similarity on 382 alarms | 0.59 – 0.75 (median 0.72); low-confidence flag 0/382 |

Text alone identifies the true class only modestly above chance (P@5 is below the majority share),
which is why the class comes from the classifier as a filter.

### 5.6 Test Suite
209 tests in 29 modules (21 unit, 8 integration): **206 pass, 3 skip** (the skipped tests need a Groq
key). A test session restores the committed ChromaDB files, which ChromaDB rewrites on open.
Appendix E lists the modules.

## 6. Discussion

### 6.1 Interpretation of Results
**RQ1.** Neither SHAP nor retrieved incidents make the LLM cite the right standard reliably: C1 5/30,
C2 7/30. **RQ2.** Yes: alignment-table grounding makes the citation correct whenever the fault is
classified correctly (19/19), lifting end-to-end correctness from 7/30 to 21/30 (+47 points). C3 matches
the copy-row rule exactly, which confirms that the curated table and the validator are what make the
citation dependable — the LLM reliably uses the standard it is given. Every remaining error comes from
Layer 2 (detection misses and misclassifications), so improving the classifier directly improves the
explanations. What the LLM adds beyond the citation (specificity, actionability, causal soundness) is
the question for Track B. **RQ3** awaits the Track B re-run.

### 6.2 G-Eval Analysis
Pending the Track B re-run (Section 5.4). This section will report per-dimension and per-condition
means, so the effect of grounding on specificity, actionability and causal soundness can be read
alongside the citation result, and the per-fault breakdown on correctly labelled windows.

### 6.3 Key Technical Findings
These were found by reviewing the system against its data, and each was fixed at its source.
1. **Retrieval only echoed the label.** `ticket_text` is one template per class (11 distinct
   texts), the Q&A's root-cause answer is the class name, and the prompt showed only the first
   200 characters — 11 different snippets across 935 tickets. With the label in the query, P@1 vs
   the true class (0.707) was below the classifier (0.748). Text alone cannot identify the fault
   (0.451 label-free), so the deployed design takes the class from the classifier as a filter,
   ranks incidents by KPI evidence, and shows the incident's own KPI narrative.
2. **The ablation measured the classifier as well as the grounding.** The first Track C sampled by
   predicted label: 8 of its 30 windows were normal traffic, 9 were misclassified, three congestion
   windows came from an older model's predictions and received another window's RCA evidence.
   Sampling by ground truth and scoring against the true fault separates the two effects.
3. **The conditions differed in their instructions.** Only C3 had the "use TR … e.g. TR 38.901"
   hint and KPI-specific field descriptions; now all three share one instruction block without
   example standard numbers.
4. **Jamming citations were scored valid.** With no alignment row, the validator accepted any
   fault's primary standard; a fault without a row now never has a valid citation, and its prompt
   says no clause applies.
5. **Truncation was not the problem.** Documents exceed MiniLM's 256-token window, but averaging
   all paragraphs retrieved far worse (0.049); the first paragraph carries the radio-layer evidence.
6. **Index metric.** An earlier index used L2 distance while similarity was computed as 1 − d; the
   index was rebuilt in cosine space (and now from scratch with `--rebuild`).
7. **Degenerate calibrator.** Farida's isotonic calibrator for Stage 1 was rejected in her pipeline
   and the file deleted; the reported confidence is uncalibrated (Section 3.4).
8. **Alignment table v1.2** moved Faulty Handover from TS 38.133 to TS 38.331 §5.5.4.4. Re-scoring
   the first run against v1.2 turned its C3 30/30 into 28/30 — one reason Track C was re-run.
9. **Dashboard data.** The dashboard browsed the Phase 1 hand-off file (935 training windows whose
   own tickets are in the index, older labels); it now browses the 382 Phase 2 test alarms.
10. **Baselines make the citation result interpretable.** A fixed-answer rule (12/30) and the
    copy-row rule (21/30) show that C3's gain comes from the alignment table, as designed, and that
    LLM-only citations (C1, C2) are not reliable without it.
11. **SHAP was shown with a misleading sign.** Layer 2 stores |SHAP|, so every prompt said "SHAP=+x",
    including 97 top-3 features that push away from the predicted class; prompts now give magnitude,
    direction and effect in words.
12. **The C3 evidence block carried a second, stale standard and test-derived labels.** RCA records
    still cite alignment v1.1 text (Handover TS 38.133) next to the table's TS 38.331, and their KPI
    "evidence status" comes from class-level statistics over labelled windows including the test split.
    Both were removed from the prompt (run v3); the citation counts did not change.
13. **Dashboard defects.** Browsing a class with one alarm (Sudden Spike) crashed the window slider,
    and a second upload displayed the first file's explanation; both fixed and covered by tests.
14. **Deployment and reproducibility.** The lock pulled CUDA builds of PyTorch (about 2 GB, no GPU used);
    `build_index.py --rebuild` without raw data silently emptied the committed index; scripts needed
    `PYTHONPATH`; the split indices were not committed. All four were fixed at the packaging level.

### 6.4 Objectives Met
| Objective | Result | Evidence |
|---|---|---|
| Detection | recall 88.7 % (83.5 % without Jamming), precision 69.6 % | 5.1, 5.2 |
| Classification | 66.3 % exact (51.5 % without Jamming; 74.8 % of detected), 75.0 % family | 5.2 |
| Attribution | top-3 SHAP with magnitude, KPI direction and effect | 3.5 |
| Citation grounding | 21/30 end to end (+47 points over C2); 19/19 when classified correctly | 5.3 |
| Explanation quality | pending G-Eval re-run | 5.4 |

### 6.5 Limitations
- **The citation metric is specification-level and the standard is in the C3 prompt.** C3 therefore
  measures how reliably the grounded system cites the right standard, not what the LLM knows unaided
  (C1 covers that) (Section 5.3).
- **Small, clustered sample.** n = 30 per condition, 3 per class, drawn from few recording sessions;
  intervals and p-values are descriptive.
- **Track A is a test-split estimate** (Section 3.4) and precision depends on the recording sessions in
  the test split; more sessions would tighten it.
- **Jamming dominates the headline accuracy:** without it, exact accuracy is 51.5 %.
- **Sudden Spike is invisible to the detector** (0 of 11 test windows flagged); congestion is mostly
  confused with Buffer Overflow. This caps end-to-end correctness for those faults whatever the LLM does.
- **C3 amplifies Layer 2 errors:** a misclassified window receives confident-looking evidence for the
  wrong class.
- **Specification-level validation only:** clause numbers are not checked.
- **G-Eval pending** on the current explanations, and an LLM judge from the generator's family needs a
  human spot-check.
- **Layer 2 is not reproducible from this branch alone:** it needs `phase2/` and its data caches; the
  model files store anonymous feature names; the training-session-only origin of the synthetic windows
  is documented but not asserted in code.
- **Data hygiene of the RCA file:** some records carry alignment v1.1 standards text and mis-encoded
  characters in fields that are not injected into prompts.
- **One LLM, one seed, free-tier quota** (15 requests/min; three 429s per run, all recovered).
- **The corpus has no incident-resolution text,** so retrieval contributes comparable KPI patterns,
  not fixes.

### 6.6 Practical Implications
For a NOC, the results show where the value is: a curated fault-to-standard table plus a validator
makes citations dependable once the fault is identified, while an ungrounded LLM often cites the wrong
specification. The next gains come from detection (Sudden Spike, congestion) and from extending the
table, and the LLM's contribution is the explanation it writes around the standard. The dashboard shows the citation check and the evidence behind each explanation, so an
engineer can see when to distrust it.

## 7. Conclusion and Future Work

### 7.1 Main Achievements
1. A session-split cascade with 88.7 % recall and 66.3 % exact accuracy over 11 fault types
   (51.5 % without Jamming).
2. An explanation pipeline whose citations are correct for the true fault in 21/30 sampled faults and
   in 19/19 correctly classified ones; all failures trace to Layer 2.
3. An evaluation that separates grounding from classification, scores against ground truth, and
   includes no-LLM baselines, reproducible from the repository (Appendix F).

### 7.2 Contributions
- A three-condition ablation with shared instructions, sampled and scored by ground truth, with
  constant and copy baselines that show what the citation metric can and cannot measure.
- A Release 19 fault-to-3GPP alignment table for the TelecomTS fault types.
- Evidence that, on TelecomTS, dense retrieval cannot substitute for the classifier, and a
  retrieval design that uses it for what it can do.

### 7.3 Future Work
- G-Eval on the 72 explanations of run v3, with a human spot-check to measure judge–human agreement.
- A citation test the prompt cannot answer: ask for the clause, or hold the table out for some faults.
- Improve detection of Sudden Spike and congestion (Layer 2), and re-estimate Track A on sessions
  that played no part in model selection.
- More windows, sessions and seeds for Track C; a second LLM.
- Reproducible Layer 2 inference inside this repository (named features, a feature builder, an
  asserted training-only synthetic-host set); regenerate `rca_evidence.json` from table v1.2.

## References
[1] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
[2] Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
[3] Liu, Y., et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. *EMNLP 2023*.
[4] Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS 2018*.
[5] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
[6] 3GPP TS 38.141-1, Base Station conformance testing, Part 1: Conducted conformance testing (Release 19).
[7] 3GPP TS 38.321, NR; Medium Access Control (MAC) protocol specification (Release 19).
[8] 3GPP TS 28.552, Management and orchestration; 5G performance measurements (Release 19).
[9] 3GPP TR 38.901, Study on channel model for frequencies from 0.5 to 100 GHz (Release 19).
[10] 3GPP TS 38.331, NR; Radio Resource Control (RRC) protocol specification (Release 19).
[11] 3GPP TS 38.104, NR; Base Station radio transmission and reception (Release 19).
[12] O-RAN Alliance. O-RAN Architecture Description (WG1).
[13] Maatouk, A., et al. (2026). TelecomTS: A Multi-Modal Telecom Dataset. *ICML 2026*. arXiv:2510.06063.
[14] Chroma. Chroma: the open-source embedding database. https://github.com/chroma-core/chroma
[15] Google. Gemini API documentation (structured output, rate limits). https://ai.google.dev

---

## Appendices

### Appendix A — 3GPP Alignment Table v1.2 (`data/alignment_table.json`)
| Fault type | 3GPP reference (Release 19) | O-RAN component | Relation |
|---|---|---|---|
| Antenna Failure | TS 38.141-1 §4.2.1 + §7.2 | O-RU | governed-by |
| Buffer Overflow (Gradual Buildup) | TS 38.321 §6.1.3.1 + §5.4.5 | O-DU | specified-in |
| Co-Channel Interference (Mild) | TS 38.141-1 §7.8 | O-RU | governed-by |
| Co-Channel Interference (Severe) | TS 38.141-1 §7.8 + TS 38.104 §4.9 | O-RU | governed-by |
| Doppler Shift (Severe) | TR 38.901 §7.6.6 + §7.6.10 | O-DU / O-RU | measured-by |
| Faulty Handover Algorithm (Too Frequent) | TS 38.331 §5.5.4.4 + §5.5.4.1 | Near-RT RIC | specified-in |
| Faulty RF Filters (Temporal) | TS 38.141-1 §7.5 + §7.6 + TS 38.104 §6.6 | O-RU | governed-by |
| High Network Congestion (Gradual Buildup) | TS 28.552 §5.1.1.2.1 + §5.1.1.2.2 | O-DU | measured-by |
| High Network Congestion (Sudden Spike) | TS 28.552 §5.1.1.2.1 + §5.1.1.2.2 | O-DU | measured-by |
| Resource Allocation Bugs | TS 38.321 §5.4.3.1.3 | O-DU / Near-RT RIC | specified-in |
| Jamming | none (external over-the-air event) | — | not-applicable |

### Appendix B — One Window under the Three Conditions
Test window 611, true and predicted fault **Doppler Shift (Severe)** (`track_c_v3/track_c_explanations.jsonl`).

**C1 — label + SHAP** · cited **TS 38.104** · correct: **no**
> Root cause: "Severe Doppler shift is induced by extremely high-speed user equipment movement,
> leading to above-normal RSRP variations and degraded DL MCS due to rapid channel decorrelation."
> O-RAN component: Near-RT RIC.

**C2 — + past incidents** · cited **TS 38.104** · correct: **no**
> Reasoning: "…heavily influenced by above-normal RSRP exhibiting sharp variability (spanning from
> -120.7 dB to -107.2 dB across similar historical traces) and suppressed downlink modulation…"
> O-RAN component: RIC xApp / gNB-DU.

**C3 — + alignment row + RCA evidence** · cited **TR 38.901** · correct: **yes**
> Root cause: "Severe Doppler shift is induced by high UE velocity, causing rapid channel phase
> variations reflected by elevated RSRP standard deviation (RSRP_std |SHAP|=0.654) and FFT energy
> alongside degraded downlink modulation (DL_MCS_mean=6)." Action: "Enable advanced channel
> estimation and robust Doppler compensation algorithms at the O-DU/O-RU to handle high-velocity UE
> trajectories." O-RAN component: O-DU / O-RU.

C2 used the retrieved incidents' concrete RSRP range; C3 cites the channel-model report it was given
(TR 38.901, as the copy rule would) and names the feature-level evidence from the RCA block.

### Appendix C — G-Eval Rubric (1–5 per dimension)
| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Citation Validity | fabricated or unrelated standard | real standard, wrong clause or overview document | specification and clause match the true fault's row |
| Fault Specificity | generic ("network issue") | right fault, generic symptoms | exact mechanism with the window's KPI directions |
| Actionability | trivial ("restart") | sensible but not localised | concrete action at the responsible O-RAN component |
| Causal Soundness | illogical chain | plausible but incomplete | cause → protocol symptom → KPI observation |

### Appendix D — Configuration (`configs/config.yaml`, excerpt)
```yaml
rag:
  embedding_model: "all-MiniLM-L6-v2"
  chroma_db_path: "data/chroma_db"
  collection_name: "telecomts_descriptions"
  top_k: 5
  cosine_threshold: 0.35
llm:
  backend: "gemini"
  gemini_model: "gemini-3.5-flash-lite"
  temperature: 0.1
  max_retries: 3
  min_request_interval_s: 2.0
  backoff_base_s: 5
  backoff_max_s: 60
data:
  train_idx_file: "train_idx_sessionsplit.npy"
  test_idx_file: "test_idx_sessionsplit.npy"
  layer2_output_file: "layer2_output_sessionsplit_phase2.json"
  alignment_table_path: "data/alignment_table.json"
  rca_evidence_path: "data/processed/rca_evidence.json"
  random_state: 42
evaluation:
  citation_validity_threshold: 0.70
  samples_per_fault: 3
models:
  model1_path: "data/models/model1.cbm"
  model2_path: "data/models/model2.cbm"
```

### Appendix E — Test Modules (`pytest --collect-only`)
| Module | Tests | Module | Tests |
|---|---:|---|---:|
| test_review3_fixes | 26 | test_schema | 10 |
| test_phase2_models | 23 | test_evaluator | 10 |
| test_llm_explainer | 23 | test_data_loader | 6 |
| test_phase2_pipeline | 18 | test_run_eval_track_c | 6 |
| test_phase2_data | 12 | test_run_pipeline | 5 |
| test_rag_query | 5 | test_run_eval_track_b | 4 |
| test_kg_indexer | 4 | test_config_loader | 3 |
| test_build_index | 3 | test_explanation_panel | 3 |
| test_run_eval_track_a | 2 | test_detection_panel / kpi_signal / shap / sources panels | 2 each |
| integration/test_01 schema | 3 | integration/test_05 llm explainer | 6 |
| integration/test_02 rag query | 4 | integration/test_06 evaluator | 4 |
| integration/test_03 chromadb | 6 | integration/test_07 full pipeline | 4 |
| integration/test_04 alignment | 6 | integration/test_08 real RAG cycle | 5 (3 skip) |
| **Total** | **209** (206 pass, 3 skip) | | |

### Appendix F — Where Each Number Comes From
| Result | File | Produced by |
|---|---|---|
| Detection, classification, retrieval metrics | `data/processed/track_a_metrics.json` | `scripts/run_eval_track_a.py` |
| Track C counts, baselines and explanations | `data/processed/track_c_v3/` (v2 for comparison) | `scripts/run_eval_track_c.py --output data/processed/track_c_v3` |
| Retrieval design comparison (0.707 / 0.102 / 0.451 / 0.049) | — | `scripts/experiment_retrieval_design.py` |
| Cascade counts (cross-check), training rows | `data/processed/validation.json`, `data/processed/log.txt` | Layer 2 pipeline (`phase2/` on main) |
| Feature list | `data/processed/feature_spec.json` | Layer 2 pipeline |
| Alignment table | `data/alignment_table.json` | Rodina Khallaf, v1.2 |
| Wilson CIs, McNemar p | computed from `track_c_v3/track_c_explanations.jsonl` | — |

