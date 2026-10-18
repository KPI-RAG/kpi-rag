# Grounded Explanations for 5G Fault Diagnosis

**Retrieval, LLM reasoning, citation checking and evaluation in the KPI-RAG system**

*Individual report · Master project, Group 15*

**Student:** Ahmed Mohamed Al Shobaki (20596291)  
**Role:** Team leader; retrieval pipeline, LLM explanation layer, evaluation, dashboard and deployment  
**Team project:** KPI-RAG: Explainable Root-Cause Analysis for 5G Networks via Standards-Grounded Retrieval-Augmented Generation  
**Supervisor:** Prof. Yuanzhu Chen  
**Program:** M.S. / Professional Master's in Computer Science (AI & Data Science), DiGiLiANS Master Program, Round 1  
**University:** Queen's University, in partnership with the National Telecommunication Institute (NTI)  
**Industry partner:** Capgemini  
**Code & demo:** [github.com/KPI-RAG/kpi-rag](https://github.com/KPI-RAG/kpi-rag/tree/master) (branch `master`) · [kpi-rag-group15.streamlit.app](https://kpi-rag-group15.streamlit.app)  

Submitted October 20, 2026

## Abstract

**Problem.** A fault classifier can tell a 5G operations engineer which fault a window of KPI measurements shows, but not why it happened or which 3GPP specification governs it. Language models write fluent explanations, yet without grounding they often cite a standard that does not apply to the fault.

**What I built.** Within the KPI-RAG team project I designed and implemented the explanation layer that runs after the fault classifier. It indexes the KPI narratives of the 935 anomalous training windows of the TelecomTS dataset in ChromaDB and, for each alarm, retrieves comparable past incidents with a query built from the KPI evidence alone, restricted to the predicted fault type. A prompt combines the classifier's SHAP evidence, the retrieved incidents, the fault's row of a 3GPP alignment table and a per-window evidence block, and Gemini 3.5 Flash Lite returns a structured JSON explanation whose citation my validator checks automatically. I also built the Streamlit dashboard, the evaluation scripts, the test suite and the packaging used to deploy the system.

**Results.** In a controlled ablation on 30 test windows sampled by true fault type, the cited standard is correct for **5/30** windows with label and SHAP only, also **5/30** with retrieved incidents, and **21/30** with the alignment row and evidence, a gain of 53 points. When the classifier's label is correct, the grounded prompt gives the right standard in **19/19** cases, and all nine remaining errors come from detection or classification upstream. A retrieval experiment showed that the dataset's tickets are one template per class, so the deployed design ranks per-window narratives inside the predicted class instead of retrieving on ticket text. The 72 LLM calls of the ablation finished with no fallback, and 207 of 210 automated tests pass.

**Conclusion.** Grounding the prompt in a curated standards table and validating the answer makes the citation dependable once the fault is known. Better upstream detection is now the main way to improve the explanations end to end.

**Keywords:** 5G, root-cause analysis, retrieval-augmented generation, 3GPP, LLM grounding, citation validation, ablation, Streamlit

## Acknowledgements

I thank my supervisor, Prof. Yuanzhu Chen, for guidance and for pushing us to evaluate what we claimed. I thank my teammates Farida El Ghabary, Raneem Alnaghy and Rodina Khallaf; their classifier output, data split, standards table and RCA records are the inputs my components consume, and their own reports describe that work. Capgemini gave us the operational view of how network operations teams troubleshoot faults. The National Telecommunication Institute (NTI) and the Ministry of Communications and Information Technology of Egypt run the DiGiLiANS Master Program, and Queen's University provided the academic programme. I also thank the authors of the TelecomTS dataset [1] for making it public.

## Table of Contents

- Abstract
- Acknowledgements
- 1 Introduction
  - 1.1 Background and motivation
  - 1.2 Problem statement
  - 1.3 Objectives and research questions
  - 1.4 Scope of this report
  - 1.5 Report organisation
- 2 Background and Literature Review
  - 2.1 Fault diagnosis in 5G and O-RAN
  - 2.2 LLMs for root-cause analysis
  - 2.3 Retrieval-augmented generation
  - 2.4 Grounding and citation checking
  - 2.5 Attribution evidence in prompts
  - 2.6 Gap addressed
- 3 Methodology
  - 3.1 Approach
  - 3.2 Data and inputs
  - 3.3 Corpus analysis and retrieval design
  - 3.4 Prompt design and the three conditions
  - 3.5 Per-window evidence block
  - 3.6 Generation settings
  - 3.7 Citation normalisation and validation
  - 3.8 Evaluation design
  - 3.9 Tools
- 4 Design and Implementation
  - 4.1 Architecture
  - 4.2 Components
  - 4.3 Data contract
  - 4.4 Error handling
  - 4.5 Dashboard
  - 4.6 Design decisions
  - 4.7 Testing, packaging and deployment
- 5 Results and Evaluation
  - 5.1 Citation ablation (Track C)
  - 5.2 Where the remaining errors come from
  - 5.3 Retrieval design experiment
  - 5.4 Reliability and testing
  - 5.5 Comparison with existing approaches
- 6 Discussion
  - 6.1 Interpretation of results
  - 6.2 Were the objectives met?
  - 6.3 Problems found and fixed at the root
  - 6.4 Practical implications
  - 6.5 Limitations
- 7 Conclusion and Future Work
- References
- Appendices A to G

**List Of Figures**

- Fig. 1: Explanation layer architecture
- Fig. 2: Citation correctness by condition
- Fig. 3: C3 outcome for the 30 windows
- Fig. 4: Retrieval design comparison

**List Of Tables**

- Tab. 1: Inputs used by the explanation layer
- Tab. 2: Prompt blocks per condition
- Tab. 3: Error handling policy
- Tab. 4: Design decisions
- Tab. 5: Track C results
- Tab. 6: Retrieval designs
- Tab. 7: Comparison with existing approaches
- Tab. 8: Objectives assessment
- Tab. A: Per-window Track C results

## 1. Introduction

### 1.1 Background and motivation

5G New Radio and Open RAN split the base station into radio, distributed and central units connected by open interfaces, which multiplies the components and KPIs an operations team has to watch. When KPIs degrade, engineers compare raw measurements with the technical specifications by hand. The work is slow and its outcome depends on who does it [4], which lengthens the mean time to repair.

A machine-learning classifier speeds up the first step by naming the fault, but an engineer still has to decide what caused it, which component is responsible and what to do. That is the step a language model can help with, provided its explanation rests on the window's evidence and cites a specification that actually governs the fault. Without grounding, the model in our ablation cited a correct standard for only 5 of 30 faults (Section 5.1), which is too unreliable for an engineer to act on.

### 1.2 Problem statement

Given a classifier alarm for one window of 5G KPI measurements (the predicted fault, its confidence, the top three SHAP features and the window's statistics), produce an explanation that names a physical root cause tied to the window's KPI values, cites the 3GPP specification and O-RAN component involved, recommends an action, and can be verified automatically. The work also has to measure how often the citation is right, and how much of the error belongs to the explanation layer and how much to the classifier in front of it.

### 1.3 Objectives and research questions

The team proposal set five objectives (O1 to O5). My part covers the last three, plus delivery of the running system:

- **P1** (proposal O3) Retrieve relevant past incidents from an index built from training data only, without leaking the label or any test window into the evidence.
- **P2** (proposal O4) Generate structured explanations whose 3GPP citation is checked automatically, with at least 70 % correct (the pre-registered threshold).
- **P3** (proposal O5) Design an evaluation that isolates the effect of grounding: a three-condition ablation sampled and scored by ground truth, with baselines that need no LLM.
- **P4** Deliver a deployable and reproducible system: dashboard, automated tests, locked dependencies and a public demo.

These lead to three research questions:

- **RQ1** Does retrieval over TelecomTS incident text add information beyond the predicted fault label?
- **RQ2** How much does standards grounding improve citation correctness compared with label and SHAP alone, and with retrieved incidents, and how much of the remaining error comes from upstream?
- **RQ3** Does the explanation layer run reliably and reproducibly under API quotas, malformed replies and configuration errors?

### 1.4 Scope of this report

This report covers my own work: everything from the classifier's output to the dashboard, and the evaluation of that part. Four inputs come from other members of the team and are used here as given (Table 1): the classifier output for the test windows, the session-based train/test split, the 3GPP alignment table v1.2 and the per-alarm RCA records. How they were built is described in my teammates' reports. Explanation-quality scoring by an LLM judge, multi-cell diagnosis and production deployment are outside this report.

### 1.5 Report organisation

Chapter 2 reviews related work. Chapter 3 explains the methodology and the evaluation design, and Chapter 4 the implementation. Chapter 5 reports the results, Chapter 6 interprets them and lists the problems I found and fixed, and Chapter 7 concludes. The appendices hold the per-window results, a sample explanation, the prompt template, the configuration, reproduction and user instructions, the test inventory and the sources of every number.

## 2. Background and Literature Review

### 2.1 Fault diagnosis in 5G and O-RAN

The O-RAN Alliance architecture [18] splits the gNB into the O-RU (RF front end, low PHY), the O-DU (high PHY, MAC, RLC, schedulers, buffers) and the O-CU (PDCP, SDAP, RRC), with the Near-RT RIC hosting control loops for mobility, traffic steering and resource allocation. A useful diagnosis names the component to act on as well as the fault, which is why every explanation in this work has an O-RAN component field.

TelecomTS [1] records windows of 128 samples at 100 ms across 18 channels from a 5G testbed. Each anomalous window comes with three pieces of text: a KPI narrative (`description`) written for that window, a troubleshooting ticket and a question-and-answer trace. These texts are the corpus my retrieval works on, and Section 3.3 shows that only the first one is specific to the window.

### 2.2 LLMs for root-cause analysis

A survey of LLMs for AIOps covers 183 papers and finds standards-grounded explanation under-explored [4]. RCACopilot uses retrieval-augmented LLMs on cloud incidents in production [5], and Roy et al. study LLM agents for root-cause analysis and how well their answers stay grounded [6]. In telecom, TN-AutoRCA builds an alarm-based benchmark with an agentic framework [7], reasoning models fine-tuned for 5G root-cause analysis outperform general models on drive-test data [8], and GALA adds graphs to LLM agent workflows [9]. None of them produces a 3GPP citation that is checked against the fault automatically.

### 2.3 Retrieval-augmented generation

Retrieval-augmented generation conditions a language model on passages fetched from an external store [10]. Passages are usually embedded with a bi-encoder such as Sentence-BERT [11]; I use `all-MiniLM-L6-v2`, a distilled MiniLM model [12] with a 256-token input window, and ChromaDB [14], which searches with HNSW approximate nearest neighbours [13]. TeleQnA shows that retrieval improves LLM accuracy on telecom-standards questions [3]. Retrieval only helps when the store holds something the model does not already have, and testing that assumption on TelecomTS changed my design (Section 5.3).

### 2.4 Grounding and citation checking

An LLM asked for a standard number produces one that looks plausible, whether or not it applies. Two techniques reduce the risk: putting the authoritative fact in the prompt, and checking the output against a known set of valid answers. I use both. The fault's row of the team's alignment table goes into the prompt, and a validator accepts the citation only if it is a well-formed TS/TR number listed in that row. Asking for JSON with fixed fields makes the check deterministic.

### 2.5 Attribution evidence in prompts

SHAP gives each feature an additive contribution to a prediction [15]. The sign of a SHAP value says whether the feature pushed toward or away from the predicted class, which is a separate question from whether the KPI was high or low, so a prompt has to state both. The classifier output in this project stores the magnitude |SHAP| and keeps the sign in another field. A prompt that printed the stored value as "SHAP=+x" was therefore wrong for every feature that pushed away from the class. Section 3.4 describes how the prompts now state magnitude, direction and effect separately.

### 2.6 Gap addressed

> **Gap.** Prior LLM-based RCA systems explain incidents but do not produce a standards citation that is verified automatically, and they are not evaluated in a way that separates the explanation step from upstream detection errors. My work builds that verification into the pipeline and measures it end to end against ground truth, with baselines that show what the measurement can and cannot detect.

## 3. Methodology

### 3.1 Approach

The explanation layer works in four steps. It indexes the KPI narratives of past incidents once, offline. For each alarm it retrieves the most similar past incidents of the predicted fault type. It then builds a prompt from the alarm's evidence, the incidents and the fault's standards row, and asks the LLM for a JSON explanation. Finally it normalises the cited standard and validates it. The evaluation runs the same code under three prompt conditions, so the effect of each kind of context can be measured on identical windows.

### 3.2 Data and inputs

TelecomTS [1] has 32,000 windows, 1,235 of them anomalous. The project splits it by recording session, so no session appears in both parts; the training part has 935 anomalous windows and the test part 6,312 windows, 300 of them faults. On the test part the classifier raises 382 alarms. My code reads the following inputs and nothing else from upstream:

| Input | File | What my code uses |
| --- | --- | --- |
| Raw dataset | `data/raw/` (TelecomTS JSONL) | label and KPI narrative of each anomalous window, for the index |
| Session split | `data/indices/*_idx_sessionsplit.npy` | which windows may be indexed (train) and evaluated (test) |
| Classifier output | `data/processed/layer2_output_sessionsplit_phase2.json` | per test window: predicted fault, confidence, top-3 SHAP entries, window statistics, ground truth (evaluation only) |
| Alignment table v1.2 | `data/alignment_table.json` | per fault: valid TS/TR numbers, clause text, O-RAN component |
| RCA records | `data/processed/rca_evidence.json` | per alarm (289, all non-Jamming alarms): SHAP attribution, observed KPI means, causal mechanism |

**Table 1.** *Inputs produced by other team members and consumed as given. Ground truth is read only by the evaluation scripts, never by the prompt.*

Only the 935 anomalous training windows are indexed, so no test window can retrieve its own narrative. Because raw records carry no identifier, a ticket's id is its position in the sorted files, and the indexer checks every ticket's label against the ground truth for that position before indexing. A missing split file stops the build instead of indexing everything.

### 3.3 Corpus analysis and retrieval design

Before settling the retrieval design I measured what each text field contains. The troubleshooting ticket has 11 distinct values across the 935 training windows, one template per fault class, and the root-cause answer in each Q&A trace is the class name. The first prototype also showed only the first 200 characters of each ticket, which gave 11 different snippets in total. Retrieving tickets therefore returned the label in another form. The KPI narrative is the only text that differs from window to window, so it is the document I index.

- **Documents.** One per training window: its own KPI narrative, embedded with `all-MiniLM-L6-v2` in a cosine-space ChromaDB collection (`telecomts_descriptions`), with the window's fault type as metadata. Narratives are longer than the model's 256-token window; I kept the truncation because averaging all paragraph embeddings retrieved far worse (Section 5.3), and the first paragraph carries the radio-layer evidence.
- **Query.** Built from the alarm alone: the top SHAP channels (one entry per channel, strongest first), whether each is above or below its normal mean, and the uplink and downlink KPI means. The fault label is not in the query.
- **Filter.** Results are restricted to incidents of the predicted fault type with a metadata filter. Text evidence on its own identifies the true class only modestly above chance, so the class comes from the classifier and retrieval ranks comparable incidents within it.
- **Output.** The top five are retrieved and the top three are shown to the LLM, 300 characters each, with their similarity. A top similarity below 0.35 is flagged as low confidence in the dashboard.

### 3.4 Prompt design and the three conditions

Every prompt is assembled from the same parts. The header gives the role, the predicted fault, the confidence and one line per SHAP feature in the form "RSRP: below normal (|SHAP|=0.42, toward predicted class)", so magnitude, KPI direction and effect on the class are all stated in words. Of the 1,146 top-3 SHAP entries in the classifier output, 97 push away from the predicted class, and the earlier "SHAP=+x" format had presented all of them as support. One instruction block, shared by all conditions, asks for exactly five JSON fields and contains no example standard numbers, since an example would hint at the answer. Appendix C gives the template.

| Block | C1 | C2 | C3 (deployed) |
| --- | --- | --- | --- |
| Header: fault, confidence, SHAP lines | yes | yes | yes |
| Per-window evidence block (Section 3.5) | no | no | yes |
| Three retrieved incidents | no | yes | yes |
| Alignment row: standard, clause, O-RAN component | no | no | yes |
| Shared instruction block (five JSON fields) | yes | yes | yes |

**Table 2.** *Content of each condition (`src/llm_explainer.py`). For Jamming, which has no alignment row, the standards block says that no 3GPP clause applies.*

### 3.5 Per-window evidence block

For the deployed condition I format the alarm's RCA record into a short block of 537 to 659 characters (`src/rca_loader.py`). It keeps the top three SHAP features with magnitude, KPI direction and effect, the observed means of the KPIs behind them, and the first 120 characters of the causal mechanism. The block leaves out two fields of the record. Its standards text is dropped because some records still carry the older v1.1 standard (Handover TS 38.133), and the alignment table must be the only source of standards in the prompt. The record's "evidence status" is dropped because it was derived from class-level statistics over labelled windows that include the test split. The block is injected only when the record belongs to the same window and fault as the alarm.

### 3.6 Generation settings

The deployed backend is Gemini 3.5 Flash Lite [16] at temperature 0.1 with JSON output. Groq and Ollama backends are also supported through the same interface. The model's answer has five fields: root cause, 3GPP reference, O-RAN component, recommended action and reasoning trace.

### 3.7 Citation normalisation and validation

The parser accepts bare JSON or JSON in a code fence and rejects replies with missing or non-string fields. It normalises the first citation it finds, so "TS38.331", "ts 38.331" and "3GPP TS 38.331 §5.5.4.4" all become `TS 38.331`. The validator then makes two checks: the reference must match the TS/TR format, and it must appear in the list of valid references for the alarm's fault. A fault with no row, such as Jamming, can never have a valid citation; an earlier version accepted any fault's primary standard in that case. A template answer produced after failed LLM calls is always marked invalid.

### 3.8 Evaluation design

#### Track C: citation ablation

Three test windows are drawn for each of the 10 fault types that have an alignment row (seed 42), sampling by true fault type rather than by predicted label. Each window goes through the full system under C1, C2 and C3. A citation counts as correct if it is valid for the window's true fault, so a misclassification that brings in the wrong row counts as wrong, and a window the detector missed has no explanation and counts as incorrect. I also record validity against the predicted fault's row, which measures how closely the LLM follows the row it was given. Two answer rules that need no LLM are scored on the same windows: always citing the single standard that is correct most often, and citing the predicted fault's row. With 30 windows per condition I report Wilson 95 % intervals [20] and exact McNemar tests [21] to describe effect sizes; they are not significance tests on a sample this small.

#### Retrieval design experiment

For the 266 test faults the classifier detected, I measure precision@1 and @5: the share of retrieved incidents whose fault type equals the window's true type. Four designs are compared: ticket documents with the label in the query (the first prototype), ticket documents without the label, KPI narratives without the label (deployed), and narratives embedded by averaging all paragraphs. The majority-class share and the accuracy of the classifier's label on these windows are the reference points.

#### Deployment and software checks

Over the 382 alarms I record the top retrieval similarity and how often the low-confidence flag fires. The automated test suite runs on every change.

### 3.9 Tools

Python 3.11 managed with uv; sentence-transformers and ChromaDB 1.5 for retrieval; the google-genai, groq and openai clients for the LLM backends; pydantic for the data contract; Streamlit 1.61 [17] for the dashboard; pytest for testing; Streamlit Community Cloud for the public demo.

## 4. Design and Implementation

### 4.1 Architecture

```text
[Inputs (from the team)] TelecomTS raw data | session split | classifier output · 6,312 windows | alignment table v1.2 | RCA records · 289
        |
        v
[Index (offline)] load + order check → train split only · 935 → MiniLM embeddings → ChromaDB · cosine
        |
        v
[Retrieve (per alarm)] label-free KPI query → filter: predicted type → top 3 incidents
        |
        v
[Explain (per alarm)] prompt builder (C1 / C2 / C3) → LLM call with retries → JSON parser → citation validator
        |
        v
[Present (& evaluate)] Streamlit dashboard · 6 panels | Track C script | retrieval experiment
```

**Figure 1.** The explanation layer. The first row lists inputs produced by other team members; everything below it is part of this work.

### 4.2 Components

#### Data loading and indexing: `src/data_loader.py`, `src/kg_indexer.py`, `scripts/build_index.py`

The loader reads the JSONL files in sorted order, keeps the anomalous records and flattens each into a ticket holding its label, ticket text, Q&A trace and narrative. `apply_train_split()` keeps only training windows and raises an error if the split file is missing. `verify_ticket_order()` compares every ticket's label with the ground truth for its position. The indexer embeds the narratives, removes duplicate ids in a batch and adds only new ids. `build_index.py --rebuild` recreates the collection from scratch and refuses to clear it when no raw tickets were loaded.

#### Retrieval: `src/rag_query.py`

`build_query()` turns an alarm into the label-free query of Section 3.3, and `query_from_classifier_output()` runs it with the predicted-type filter. Similarity is computed as one minus the cosine distance, and the function returns the incidents together with a low-confidence flag.

#### Explanation: `src/llm_explainer.py`, `src/rca_loader.py`

The explainer builds the prompts from the shared parts, calls the configured backend through a shared rate limiter, parses and normalises the reply and validates the citation. `explain()` is the deployed path (C3), and `explain_condition()` exposes all three conditions for the ablation. The RCA loader indexes the records by window and formats the evidence block.

#### Pipeline and evaluation: `scripts/run_pipeline.py`, `scripts/run_eval_track_c.py`, `scripts/experiment_retrieval_design.py`

The pipeline script explains one alarm from the command line. The Track C script samples by ground truth, runs the three conditions, scores them together with the two no-LLM rules and writes the explanations, a summary and a scoring template. It refuses to overwrite an earlier run without `--force`, and `--summarise-only` recomputes the summary from saved explanations without new LLM calls. The retrieval experiment reproduces Table 6. A further script, `run_eval_track_a.py`, recomputes the classifier's metrics from its output file, which I used to check that the file I consume matches the ML team's own validation counts.

### 4.3 Data contract

All components exchange pydantic models (`src/schema.py`), so a malformed input fails at the boundary instead of deep inside a prompt.

- `ClassifierOutput`: the fault type, which must be one of the 11 known types; a confidence between 0 and 1; exactly three SHAP entries, each with channel, value, feature name, direction against the normal mean and effect on the class; and the window statistics. The older field name `predicted_fault_type` is accepted as an alias.
- `RetrievedTicket`: id, narrative, fault type and similarity.
- `LLMExplanation`: the five answer fields plus `reference_valid`, `template_generated` and `fallback_reason`, so the dashboard and the evaluation can tell a model answer from a template and see why the template was used.

### 4.4 Error handling

The ablation and the public demo both run on a free-tier API quota, so the client has to separate errors worth retrying from errors that a retry cannot fix.

| Situation | Behaviour |
| --- | --- |
| Every call | at least 2 s after the previous one, for all backends |
| Rate limit (429) | wait for the server's Retry-After, or back off exponentially from 5 s with jitter; give up if the server asks for more than 60 s, as with a daily quota |
| Configuration or client error (missing key, 400/401/403/404) | stop at once; retrying the same request cannot succeed |
| Malformed or incomplete JSON | retry |
| Other transient errors (timeouts, 5xx) | retry after 1, 2, 4 ... s, at most 10 s |
| Three attempts used | template explanation with its cause in `fallback_reason`, always counted as invalid |

**Table 3.** *Retry policy in `_run_with_retry()` and `_is_non_retryable()`.*

### 4.5 Dashboard

The Streamlit dashboard (`dashboard/app.py`, six components in `dashboard/components/`) shows one alarm at a time in six panels:

1. **Detection**: predicted fault and confidence;
2. **SHAP**: the top contributing KPIs;
3. **KPI window statistics**;
4. **Explanation**: root cause, action and reasoning, with a ✅/❌ citation badge and the cause when a template was used;
5. **Sources**: the retrieved incidents with their similarity, flagged when weakly similar;
6. **RCA evidence**: the standard and O-RAN component for the fault and the evidence injected into the prompt.

The input can be a built-in example, an uploaded `ClassifierOutput` JSON, or one of the 382 test alarms, filtered by fault type. Since these are all test windows, none of them has its own narrative in the index. Uploads are keyed by content, so a new file never shows an old explanation, and the dashboard asks for the API key of whichever backend is configured.

### 4.6 Design decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| Retrieval documents | per-window KPI narratives | ticket text is one template per class (Section 3.3) |
| Fault label | metadata filter, not part of the query | text alone cannot identify the class; a label in the query only echoes it |
| Index metric | cosine space, rebuilt from scratch | an early L2 index made "1 minus distance" a meaningless similarity |
| Standards source | alignment table only | a second source (RCA records) produced conflicting references after the table was updated |
| Instructions | one block for all conditions, no example standards | conditions must differ only in context, and an example hints the answer |
| Ablation sampling | by true fault, scored against the true row | separates grounding from classifier errors |
| Failure handling | fail closed; labelled template | a missing split or a broken key should stop the run instead of producing plausible output |

**Table 4.** *Main design decisions in the explanation layer.*

### 4.7 Testing, packaging and deployment

The test suite has 210 tests in 29 modules: 21 unit modules and 8 integration modules. They cover the data contract, index building, retrieval, prompt construction for each condition, parsing and citation validation, the retry policy (with mocked rate limits and client errors), the Track C sampling and scoring, every dashboard panel and a headless run of the whole app. A test fixture restores the committed index after tests that modify it.

Dependencies are locked with uv, with CPU-only PyTorch: the default Linux wheel pulled about 2 GB of CUDA libraries although no GPU is used. `requirements.txt` is exported from the same lock for Streamlit Community Cloud. The source folders are installed as a package, so scripts run without setting `PYTHONPATH`, and the split indices and the 17 MB index are committed so a fresh clone can run the tests and the dashboard at once. The public demo deploys from branch `master`.

## 5. Results and Evaluation

### 5.1 Citation ablation (Track C)

Run v3 (`data/processed/track_c_v3/`) covers 30 windows, 3 per true fault type. Of these, the classifier labels 19 correctly, misclassifies 5 and misses 6, which leaves 24 windows with an explanation and 72 LLM calls in total. No call fell back to the template. I repeated the run after the team updated the RCA evidence file. C3 was identical in every window, and the C1 and C2 differences from the previous run are run-to-run variation of the LLM, because their prompts contain no RCA evidence.

| Answer rule | Correct (n = 30) | Wilson 95 % CI | Classified right (n = 19) | Valid vs predicted (n = 24) |
| --- | --- | --- | --- | --- |
| C1 LLM: label + SHAP | 5/30 (16.7 %) | 7.3-33.6 % | 4/19 | 7/24 |
| C2 LLM: + past incidents | 5/30 (16.7 %) | 7.3-33.6 % | 4/19 | 7/24 |
| C3 LLM: + alignment row + evidence | **21/30 (70.0 %)** | 52.1-83.3 % | **19/19** | 24/24 |
| No LLM: always cite TS 38.141-1 | 12/30 (40.0 %) | — | — | — |
| No LLM: cite the predicted fault's row | 21/30 (70.0 %) | — | 19/19 | 24/24 |

**Table 5.** *Track C (`track_c_v3/track_c_summary.json`). "Correct" is scored against the true fault's row; "valid vs predicted" against the row of the fault the classifier named, over the 24 windows with an explanation. Per-window outcomes are in Appendix A.*

**Citation correct for the true fault, out of 30 windows.** LLM conditions and answer rules that need no LLM.

| Series | Value |
|---|---|
| C1 · label + SHAP | 5 / 30 |
| C2 · + past incidents | 5 / 30 |
| C3 · + alignment + evidence | 21 / 30 |
| Always TS 38.141-1 | 12 / 30 |
| Cite predicted row | 21 / 30 |

**Figure 2.** Grounding raises end-to-end citation correctness from 5/30 to 21/30 (+53 points) and reaches the pre-registered threshold.

When the correct standard is in the prompt, which is the case for the 19 correctly classified windows, the LLM cites it in 19/19 (Wilson 83.2-100 %). Moving from C2 to C3 improves 16 windows and worsens none (exact McNemar p = 3.1·10⁻⁵), and moving from C1 to C3 also improves 16 and worsens none (p = 3.1·10⁻⁵).

C3 gets the same score as the rule that cites the predicted fault's row (21/30). In all 24 explained windows the LLM adopted the standard it was given, which is what the design intends: the standard comes from the curated table and is held to it by the validator, while the model writes the explanation around it. C3's end-to-end accuracy therefore follows the classifier's accuracy.

Without the table the LLM's citations are unreliable, and both C1 and C2 score below the fixed answer. C1 cites TS 38.104 (12 windows), TS 38.331 (8) and TS 38.214 (4). Retrieved incidents change no window: C2 cites the same standard as C1 in all 24 explained windows, and both score 5/30.

### 5.2 Where the remaining errors come from

**What happened to each of the 30 windows under C3.** Every C3 error starts before the explanation layer.

| Outcome | Windows |
|---|---|
| classified correctly, citation correct | 19 |
| misclassified, but the wrong row cites the same specification | 2 |
| misclassified, wrong row injected | 3 |
| missed by the detector, no explanation | 6 |

**Figure 3.** The six misses are all three Sudden Spike samples, one Congestion (Gradual), one Buffer Overflow and one Handover window. The three wrong rows come from Congestion predicted as Buffer Overflow (2) and Handover predicted as RF Filters (1).

The two misclassified windows that still score correctly are CCI windows predicted as RF Filters; both rows list TS 38.141-1. Appendix B shows one window (611, Doppler Shift) under the three conditions: C1 and C2 cite TS 38.104, while C3 cites TR 38.901 and quotes the window's evidence (RSRP_std |SHAP| = 0.654, DL_MCS_mean = 6).

### 5.3 Retrieval design experiment

| Design | P@1 | P@5 |
| --- | --- | --- |
| Ticket documents, label in the query (first prototype) | 0.707 | 0.686 |
| Ticket documents, label removed | 0.102 | 0.069 |
| KPI narratives, label-free query (deployed ranking) | 0.451 | 0.335 |
| KPI narratives, all paragraphs averaged | 0.049 | 0.201 |

**Table 6.** *Share of retrieved incidents with the window's true fault type, over the 266 detected test faults (`scripts/experiment_retrieval_design.py`). Reference points: majority-class share 0.353; the classifier's label is correct for 0.748 of these windows.*

**Does the retrieved incident have the window's true fault type? (precision@1).** Reference points: majority-class share 0.353 and the classifier's label accuracy 0.748.

| Series | Value |
|---|---|
| Tickets · label in query | 0.707 |
| Tickets · label-free | 0.102 |
| Narratives · label-free | 0.451 |
| Narratives · averaged | 0.049 |

**Figure 4.** With the label in the query, retrieval repeated the label and still scored below the label itself. Without it, ticket text carries almost no signal, and narratives are only modestly above the majority share.

Retrieval over this corpus cannot replace the classifier's label, so the deployed design uses the label as a filter. Within the class, the narrative is the only text that describes the particular incident. Ranking by narrative gives the LLM KPI patterns from comparable incidents, where ticket retrieval would return the same template three times. In deployment, the top retrieved incident's similarity over the 382 alarms ranges from 0.59 to 0.75 (median 0.72), and the low-confidence flag never fired.

### 5.4 Reliability and testing

All 72 ablation calls returned a valid JSON answer without a template fallback, under the rate limiter and the free-tier quota. On the current code, **207 of 210 tests pass**; the 3 skipped tests need a Groq API key. A clean clone runs the tests and the dashboard after `uv sync`, without extra setup.

### 5.5 Comparison with existing approaches

| System | Domain | How the explanation is produced | Standard cited and checked |
| --- | --- | --- | --- |
| RCACopilot [5] | cloud incidents | retrieval-augmented LLM over past incidents | not addressed |
| Roy et al. [6] | cloud incidents | LLM agents with tools | not addressed |
| TN-AutoRCA [7] | telecom alarms | agentic LLM framework | not addressed |
| Sana et al. [8] | 5G drive-test data | fine-tuned reasoning LLMs | not addressed |
| Ungrounded LLM (C1, this study) | 5G KPI windows | label and SHAP evidence only | 5/30 correct |
| This work (C3) | 5G KPI windows | retrieved incidents, alignment row, per-window evidence | 21/30 correct; 19/19 when the label is right; checked automatically |

**Table 7.** *Qualitative comparison. The published systems use other data and tasks, so only the C1 row is a like-for-like baseline.*

## 6. Discussion

### 6.1 Interpretation of results

**RQ1.** No, not on this corpus. The tickets are class templates, so retrieving them adds nothing the label does not say, and the narratives identify the class only modestly above chance. Retrieval is still useful for a different job: once the class is fixed, it supplies concrete KPI patterns from comparable incidents, such as the RSRP values from similar incidents in C2's reasoning for window 611. It does not improve the citation (C1 and C2 both 5/30).

**RQ2.** Grounding raises citation correctness from 5/30 to 21/30, and to 19/19 when the label is right. The match with the copy-row rule shows that the improvement comes from the curated table reaching the prompt and the validator holding the answer to it. All nine remaining errors start upstream: six windows were never flagged and three were given the wrong fault. Improving the classifier is now the most direct way to improve the explanations.

**RQ3.** Yes, at the scale tested. The ablation ran without a fallback, errors that cannot be fixed by retrying stop the run with a clear cause, and the test suite and committed artifacts let a fresh clone reproduce the system.

### 6.2 Were the objectives met?

| Objective | Evidence | Status |
| --- | --- | --- |
| P1 Retrieval without leakage | train-only index with order check; label-free query; measured that text alone cannot identify the fault, so the class is a filter | Partly met |
| P2 Checked citations, at least 70 % correct | 21/30 = 70.0 % end to end; 19/19 when classified correctly | Met |
| P3 Ablation that isolates grounding | ground-truth sampling, shared instructions, two no-LLM baselines, per-window results | Met |
| P4 Deployable, reproducible system | public dashboard; 206/210 tests pass; locked CPU-only dependencies; committed artifacts | Met |

**Table 8.** *My objectives, mapped to the proposal's O3 to O5.*

### 6.3 Problems found and fixed at the root

Most of the work after the first prototype went into finding cases where the system produced plausible output for the wrong reason. Each fix below changed the component where the problem started.

1. **Retrieval only echoed the label.** The prototype indexed ticket text, which is one template per class, put the label in the query and showed 200 characters per ticket. It now indexes narratives, filters by class and ranks by KPI evidence (Section 3.3).
2. **The first ablation measured the classifier as well as the grounding.** It sampled by predicted label: 8 of its 30 windows were normal traffic, 9 were misclassified, and three congestion windows came from an older model's predictions and received another window's evidence. The current run samples by true fault and scores against the true row.
3. **The conditions had different instructions.** Only C3 carried a "use TR …, e.g. TR 38.901" hint and KPI-specific field descriptions. All three now share one instruction block with no example standards.
4. **Jamming citations were scored as valid.** With no alignment row, the validator accepted any fault's primary standard. A fault without a row can no longer have a valid citation, and its prompt says no clause applies.
5. **The index used the wrong metric.** An early index used L2 distance while similarity was computed as one minus the distance. It is now built in cosine space from scratch.
6. **A table update changed the first run's score.** When the alignment table moved Handover from TS 38.133 to TS 38.331, re-scoring the first run turned its C3 result from 30/30 into 28/30. This was one of the reasons for re-running the ablation on the current table.
7. **SHAP was shown with a misleading sign.** The prompts printed the stored magnitude as "SHAP=+x", including for the 97 features that push away from the class. They now state magnitude, direction and effect in words.
8. **The evidence block carried a stale standard and test-derived labels.** Both fields were removed from the prompt in run v3 (Section 3.5), and the citation counts did not change.
9. **The dashboard browsed the wrong windows.** It listed 935 training windows whose own narratives were in the index. It now lists the 382 test alarms.
10. **Two dashboard defects.** Browsing a class with a single alarm (Sudden Spike) crashed the window slider, and a second upload displayed the first file's explanation. Both are fixed and covered by tests.
11. **Deployment and reproducibility.** The lock pulled CUDA builds of PyTorch, a rebuild without raw data silently emptied the committed index, scripts needed `PYTHONPATH`, and the split indices were not committed. All four were fixed in the packaging.

### 6.4 Practical implications

For a network operations team, the result is that an LLM explanation can carry a dependable standards citation if the standard comes from a curated table and the answer is checked against it. The model's own choice of standard is not dependable: C1 and C2 score below always giving the same answer. The dashboard shows the citation check, the retrieved precedents and the evidence that went into the prompt, so an engineer can see why an explanation says what it says and can tell a template answer from a model answer.

### 6.5 Limitations

- **The citation metric is specification-level, and in C3 the standard is in the prompt.** Clause numbers are not checked. C3 measures how reliably the grounded system cites the right standard; C1 measures what the model knows unaided.
- **Small, clustered sample.** 30 windows per condition, 3 per class, drawn from a small number of recording sessions. The intervals and p-values are descriptive.
- **Dependence on the inputs.** End-to-end correctness is bounded by the classifier and by the alignment table. A misclassified window receives confident-looking evidence for the wrong fault.
- **Explanation quality is not scored here.** The ablation checks the citation; how specific, actionable and causally sound the text is has not been measured in this report.
- **One model, one seed, free tier.** All runs use Gemini 3.5 Flash Lite at one seed. The model name is accepted by the API but is not in Google's public model list, so the configuration names two fallbacks.
- **No resolution text in the corpus.** Retrieved incidents contribute comparable KPI patterns, not records of past fixes.

## 7. Conclusion and Future Work

### 7.1 Main achievements

1. An explanation layer that turns a classifier alarm into a structured explanation with an automatically checked 3GPP citation, deployed as a public six-panel dashboard.
2. A retrieval design based on a measured analysis of the corpus: a train-only narrative index, a label-free query and a class filter.
3. Citations correct for the true fault in 21 of 30 sampled faults and in 19 of 19 correctly classified ones, which meets the pre-registered 70 % threshold.
4. An evaluation that separates grounding from upstream errors, with no-LLM baselines, and a reproducible codebase with 210 automated tests.

### 7.2 Contributions

- **Method:** a three-condition citation ablation sampled and scored by ground truth, with two answer rules that need no LLM, which shows where a citation gain comes from.
- **Empirical finding:** on TelecomTS, retrieval over ticket text cannot substitute for the classifier, because the tickets are class templates; per-window narratives are the only incident-specific text.
- **Engineering:** a fail-closed explanation pipeline with a typed data contract, a retry policy that separates fixable from unfixable errors, and a deployment that a fresh clone reproduces.

### 7.3 Future work

- Score explanation quality (specificity, actionability, causal soundness) on the 72 saved explanations, with human raters as a check.
- Validate citations at clause level, and add a test the prompt cannot answer, for example by holding the table back for some faults.
- Repeat the ablation with more windows, sessions and seeds, and with a second LLM.
- Add resolution notes to the corpus, so retrieval can contribute past fixes as well as KPI patterns.
- Run the classifier inside the same service, so the dashboard can explain new windows directly.

## References

[1] A. Feng, A. Varvarigos, I. Panitsas, D. Fernandez, J. Wei, Y. Guo, J. Chen, A. Maatouk, L. Tassiulas, and R. Ying, “TelecomTS: A multi-modal observability dataset for time series and language analysis,” arXiv:2510.06063, Oct. 2025.

[2] A. Feng *et al.*, “TelecomTS,” Hugging Face dataset, 2026. [Online]. Available: https://huggingface.co/datasets/AliMaatouk/TelecomTS

[3] A. Maatouk, F. Ayed, N. Piovesan, A. De Domenico, M. Debbah, and Z.-Q. Luo, “TeleQnA: A benchmark dataset to assess large language models telecommunications knowledge,” *IEEE Network*, 2025.

[4] L. Zhang, T. Jia, M. Jia, Y. Wu, A. Liu, Y. Yang, Z. Wu, X. Hu, P. Yu, and Y. Li, “A survey of AIOps in the era of large language models,” *ACM Comput. Surv.*, 2025, doi: 10.1145/3746635.

[5] Y. Chen *et al.*, “Automatic root cause analysis via large language models for cloud incidents,” in *Proc. EuroSys*, 2024.

[6] D. Roy, X. Zhang, R. Bhave, C. Bansal, P. Las-Casas, R. Fonseca, and S. Rajmohan, “Exploring LLM-based agents for root cause analysis,” in *Proc. FSE*, 2024.

[7] K. Wu, Q. Yu, M. Mei, R. Liu, J. Wang, K. Zhang, and Y. Bao, “TN-AutoRCA: Benchmark construction and agentic framework for self-improving alarm-based root cause analysis in telecommunication networks,” arXiv:2507.18190, Jul. 2025.

[8] M. Sana, N. Piovesan, A. De Domenico, Y. Kang, H. Zhang, M. Debbah, and F. Ayed, “Reasoning language models for root cause analysis in 5G wireless networks,” arXiv:2507.21974, Jul. 2025.

[9] Y. Tian, Y. Liu, Z. Chong, Z. Huang, and H.-A. Jacobsen, “GALA: Can graph-augmented large language model agentic workflows elevate root cause analysis?,” arXiv:2508.12472, 2025.

[10] P. Lewis *et al.*, “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in *Proc. NeurIPS*, 2020.

[11] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in *Proc. EMNLP-IJCNLP*, 2019.

[12] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, “MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers,” in *Proc. NeurIPS*, 2020.

[13] Y. A. Malkov and D. A. Yashunin, “Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 42, no. 4, pp. 824–836, 2020.

[14] Chroma, “Chroma: The open-source embedding database.” [Online]. Available: https://github.com/chroma-core/chroma

[15] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Proc. NeurIPS*, 2017.

[16] Google, “Gemini API documentation.” [Online]. Available: https://ai.google.dev

[17] Streamlit, “Streamlit documentation.” [Online]. Available: https://docs.streamlit.io

[18] O-RAN Alliance, “O-RAN architecture description,” O-RAN.WG1 technical specification.

[19] 3GPP TS 38.141-1, TS 38.104, TS 38.321, TS 38.331, TS 28.552 and TR 38.901, Release 19, 3rd Generation Partnership Project.

[20] E. B. Wilson, “Probable inference, the law of succession, and statistical inference,” *J. Amer. Statist. Assoc.*, vol. 22, no. 158, pp. 209–212, 1927.

[21] Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.

## Appendices

### Appendix A: per-window Track C results (run v3)

| Window | True fault | Predicted | C1 | C2 | C3 |
| --- | --- | --- | --- | --- | --- |
| 597 | CCI (Mild) | CCI (Mild) | ✗ 38.214 | ✗ 38.214 | ✓ 38.141-1 |
| 592 | CCI (Mild) | CCI (Mild) | ✗ 38.214 | ✗ 38.214 | ✓ 38.141-1 |
| 645 | CCI (Mild) | RF Filters | ✗ 38.104 | ✗ 38.104 | ✓ 38.141-1 |
| 835 | Buffer Overflow | Buffer Overflow | ✗ 38.331 | ✗ 38.331 | ✓ 38.321 |
| 893 | Buffer Overflow | Buffer Overflow | ✗ 38.331 | ✗ 38.331 | ✓ 38.321 |
| 717 | Buffer Overflow | Normal (missed) | ✗ — | ✗ — | ✗ — |
| 703 | CCI (Severe) | RF Filters | ✓ 38.104 | ✓ 38.104 | ✓ 38.141-1 |
| 607 | CCI (Severe) | CCI (Severe) | ✗ 38.214 | ✗ 38.214 | ✓ 38.141-1 |
| 700 | CCI (Severe) | CCI (Severe) | ✗ 38.214 | ✗ 38.214 | ✓ 38.141-1 |
| 864 | Antenna Failure | Antenna Failure | ✗ 38.104 | ✗ 38.104 | ✓ 38.141-1 |
| 622 | Antenna Failure | Antenna Failure | ✗ 38.104 | ✗ 38.104 | ✓ 38.141-1 |
| 687 | Antenna Failure | Antenna Failure | ✗ 38.104 | ✗ 38.104 | ✓ 38.141-1 |
| 697 | RF Filters | RF Filters | ✓ 38.104 | ✓ 38.104 | ✓ 38.141-1 |
| 610 | RF Filters | RF Filters | ✓ 38.104 | ✓ 38.104 | ✓ 38.141-1 |
| 605 | RF Filters | RF Filters | ✓ 38.104 | ✓ 38.104 | ✓ 38.141-1 |
| 682 | Congestion (Gradual) | Normal (missed) | ✗ — | ✗ — | ✗ — |
| 845 | Congestion (Gradual) | Buffer Overflow | ✗ 38.331 | ✗ 38.331 | ✗ 38.321 |
| 846 | Congestion (Gradual) | Buffer Overflow | ✗ 38.331 | ✗ 38.331 | ✗ 38.321 |
| 713 | Doppler Shift | Doppler Shift | ✗ 38.104 | ✗ 38.104 | ✓ TR 38.901 |
| 611 | Doppler Shift | Doppler Shift | ✗ 38.104 | ✗ 38.104 | ✓ TR 38.901 |
| 658 | Doppler Shift | Doppler Shift | ✗ 38.104 | ✗ 38.104 | ✓ TR 38.901 |
| 648 | Faulty Handover | RF Filters | ✗ 38.104 | ✗ 38.104 | ✗ 38.141-1 |
| 879 | Faulty Handover | Normal (missed) | ✗ — | ✗ — | ✗ — |
| 715 | Faulty Handover | Faulty Handover | ✓ 38.331 | ✓ 38.331 | ✓ 38.331 |
| 684 | Resource Alloc. Bugs | Resource Alloc. Bugs | ✗ 38.331 | ✗ 38.331 | ✓ 38.321 |
| 710 | Resource Alloc. Bugs | Resource Alloc. Bugs | ✗ 38.331 | ✗ 38.331 | ✓ 38.321 |
| 686 | Resource Alloc. Bugs | Resource Alloc. Bugs | ✗ 38.331 | ✗ 38.331 | ✓ 38.321 |
| 866 | Congestion (Sudden Spike) | Normal (missed) | ✗ — | ✗ — | ✗ — |
| 827 | Congestion (Sudden Spike) | Normal (missed) | ✗ — | ✗ — | ✗ — |
| 843 | Congestion (Sudden Spike) | Normal (missed) | ✗ — | ✗ — | ✗ — |
|  | Total correct |  | 5 / 30 | 5 / 30 | 21 / 30 |

**Table A.** *Cited specification per condition (TS prefix omitted unless TR) and whether it is valid for the true fault. "—": the detector missed the window, so no explanation was produced.*

### Appendix B: one window explained under the three conditions

Test window 611, true and predicted fault Doppler Shift (Severe) (`track_c_v3/track_c_explanations.jsonl`).

**C1 · label + SHAP · cites TS 38.104** (incorrect)

> "Severe Doppler shift is caused by rapid user equipment movement resulting in abnormally high RSRP combined with degraded downlink modulation and coding scheme (DL_MCS: below normal)."

O-RAN component: Near-RT RIC

**C2 · + past incidents · cites TS 38.104** (incorrect)

> "Severe Doppler shift is induced by high-velocity user equipment mobility causing rapid radio channel fluctuations, evidenced by fluctuating RSRP levels above normal combined with suppressed downlink modulation and coding (DL_MCS below normal)."

O-RAN component: Near-RT RIC

**C3 · + alignment row + evidence · cites TR 38.901** (correct)

> "Severe Doppler shift caused by high user equipment velocity resulting in rapid channel phase variation, evidenced by high RSRP variance (std |SHAP|=0.654, fft energy |SHAP|=0.510) and degraded downlink modulation (DL_MCS mean=6, |SHAP|=0.484)."

Recommended action: "Enable high-speed mobility enhancements and adjust channel estimation filter lengths at the O-DU/O-RU to compensate for rapid phase rotation." O-RAN component: O-RU / O-DU

C1 and C2 describe the same pattern and cite the same specification; C2's reasoning adds RSRP values from the similar incidents, but this does not change its citation. C3 cites the channel-model report that governs Doppler and quotes the feature-level evidence from the evidence block.

### Appendix C: prompt template (`src/llm_explainer.py`)

Header, used by all conditions:

```
You are a 5G network fault diagnosis expert.

Fault detected: {fault type}
Confidence: {confidence:.0%}

Top contributing KPIs (SHAP):
  {channel}: {above|below} normal (|SHAP|={value:.2f}, {toward|away from} predicted class)
  ... (three lines)
```

Blocks added by C2 and C3 (C3 also adds `[EVIDENCE FROM RCA PIPELINE]`, Section 3.5):

```
Retrieved similar incidents:
  [1] (similarity 0.72) {first 300 characters of the incident's KPI narrative}...
  [2] ...
  [3] ...

[STANDARDS REFERENCE]                        (C3 only)
3GPP {standard} clause {clause}: {evidence text}
O-RAN component: {component}
```

Instruction block, identical in all conditions:

```
Ground your explanation in the specific KPI values and SHAP evidence provided above.

Return ONLY a JSON object with exactly these fields:
{
  "root_cause": "one sentence physical explanation referencing specific KPI values",
  "3gpp_reference": "TS XX.XXX or TR XX.XXX",
  "oran_component": "component name",
  "recommended_action": "one actionable step",
  "reasoning_trace": "2-3 sentence causal chain citing specific KPI values from the evidence above"
}
```

### Appendix D: configuration (`configs/config.yaml`, excerpt)

```
rag:
  embedding_model: "all-MiniLM-L6-v2"
  collection_name: "telecomts_descriptions"   # per-window KPI narratives, cosine
  top_k: 5
  cosine_threshold: 0.35
llm:
  backend: "gemini"                           # gemini | groq | ollama
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
```

### Appendix E: reproduction and user instructions

#### Install and test

```
git clone -b master https://github.com/KPI-RAG/kpi-rag.git
cd kpi-rag
uv sync                      # CPU-only torch
cp .env.example .env         # add GEMINI_API_KEY
uv run pytest tests/ -q      # 207 passed, 3 skipped
```

#### Reproduce the reported numbers

```
uv run python scripts/run_eval_track_c.py --output data/processed/my_run   # 72 LLM calls
uv run python scripts/run_eval_track_c.py --output data/processed/track_c_v3 --summarise-only
uv run python scripts/experiment_retrieval_design.py                       # needs data/raw (TelecomTS)
uv run python scripts/build_index.py --rebuild                             # rebuild the index from data/raw
```

#### Using the dashboard

```
uv run streamlit run dashboard/app.py --server.fileWatcherType none
```

1. Choose an input in the sidebar: *Use example*, *Browse Dataset Windows* (filter by fault type, then pick a window) or *Upload JSON*.
2. Click **Generate Explanation**.
3. Read the diagnosis and its citation badge, then check the retrieved incidents and the evidence panel before acting.

### Appendix F: test inventory

| Module | Tests | Module | Tests |
| --- | --- | --- | --- |
| test_review3_fixes | 26 | test_schema | 10 |
| test_phase2_models | 23 | test_evaluator | 10 |
| test_llm_explainer | 23 | test_data_loader | 6 |
| test_phase2_pipeline | 18 | test_run_eval_track_c | 6 |
| test_phase2_data | 12 | test_run_pipeline | 5 |
| test_rag_query | 5 | test_run_eval_track_b | 4 |
| test_kg_indexer | 4 | test_config_loader | 3 |
| test_build_index | 3 | test_explanation_panel | 3 |
| test_run_eval_track_a | 2 | detection / KPI / sources panels; SHAP panel | 2 each; 3 |
| integration · schema, RAG query, ChromaDB, alignment | 3 · 4 · 6 · 6 | integration · explainer, evaluator, pipeline, live RAG | 6 · 4 · 4 · 5 |
| Total | 210 | passing / skipped (Groq key) | 207 / 3 |

### Appendix G: sources of the reported numbers

| Result | Source |
| --- | --- |
| Track C counts, baselines, intervals | `data/processed/track_c_v3/track_c_summary.json` |
| Per-window explanations and citations | `data/processed/track_c_v3/track_c_explanations.jsonl` |
| Retrieval design comparison (Table 6) | `scripts/experiment_retrieval_design.py` |
| Ticket and narrative analysis (Section 3.3) | `src/kg_indexer.py`, `scripts/experiment_retrieval_design.py` |
| Test counts | `uv run pytest tests/ -q`, `pytest --collect-only` |

---

*Individual report · Ahmed Mohamed Al Shobaki · KPI-RAG, Group 15 · October 2026. Source: docs/Individual Reports/KPI_RAG_Individual_Report_Ahmed.md in the project repository (branch master).*
