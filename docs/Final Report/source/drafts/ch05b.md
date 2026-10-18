### 5.6 Retrieval design

The first prototype retrieved the top five troubleshooting tickets with the predicted label in the query, and condition C2 of the first ablation received them. Because the tickets are one template per fault class (Section 3.10), and the prototype's 200-character snippets gave only 11 different texts, this retrieval returned the label in another form.

The retrieval experiment of Section 3.11 measured what retrieval can contribute over the 266 test faults that the detector flags. It used two reference points: the majority-class share (0.353) and the accuracy of the classifier's own label on the same windows (0.748). {tab:retrieval} lists the four designs and {fig:retrieval} plots precision@1 against the two reference lines.

Table: Without the label in the query, only the deployed KPI-narrative ranking retrieves incidents of the true fault type above the majority-class share (0.353), and no design reaches the classifier's label accuracy (0.748). Values are shares of retrieved incidents with the window's true fault type, over the 266 detected test faults. {#tab:retrieval}

| Design | Precision@1 | Precision@5 |
| --- | --- | --- |
| Ticket documents, label in the query (first prototype) | 0.707 | 0.686 |
| Ticket documents, label removed | 0.102 | 0.069 |
| KPI narratives, label-free query (deployed ranking) | 0.451 | 0.335 |
| KPI narratives, all paragraphs averaged | 0.049 | 0.201 |

![With the label in the query, retrieval repeats the label and still scores below the label itself (0.748). Without it, ticket text falls below the majority share (0.353), and KPI narratives reach 0.451, modestly above it (precision@1 over the 266 detected test faults).](figures/fig_retrieval_design.png){#fig:retrieval}

With the label in the query, ticket retrieval scored 0.707, below the 0.748 of the label it was echoing. Without the label, ticket text carried almost no signal (0.102, below the majority share). Narratives with a label-free query reached 0.451, modestly above the majority share. Averaging the embeddings of all narrative paragraphs performed far worse (0.049), which is why the deployed index keeps the narrative truncated to the embedding model's input window (Section 3.10).

Retrieval over this corpus cannot replace the classifier's label, so the final design ranks KPI narratives within the predicted fault type, selected by a metadata filter (Section 3.10). Over the 382 alarms of the test sessions, the top retrieved incident's similarity ranges from 0.59 to 0.75 (median 0.72), and the low-confidence flag (similarity below 0.35) never fired.

### 5.7 Citation grounding: Track C from the first ablation to the final run

Track C addresses RQ2: does grounding in the alignment table (matrix v1.2) raise citation faithfulness over ticket-only and ungrounded generation? The ablation was first run on the prototype and later repeated under a revised protocol.

**First ablation.** The first prototype ran the three-condition ablation over 30 test windows, 3 for each of the 10 fault types with a 3GPP reference (Jamming excluded), drawn with `random.seed(42)`. C1 gave the LLM the fault type label and signal statistics; C2 added the top five retrieved tickets; C3 added the RCA evidence and the alignment table row. A citation counted as valid if it passed the format check and appeared in the alignment row of the predicted fault type. C1 produced 2/30 valid citations (7 %), C2 3/30 (10 %) and C3 27/30 (90 %). C2 and C3 each had one template fallback (Section 5.10), counted as invalid. The Phase 1 report read the result as a gain of 80 percentage points (pp) from C2 to C3 that passed the O4 criterion of 70 %. It also noted that C3 measured prompt compliance: whether the LLM uses the correct standard when it is given in the prompt.

**Why the protocol changed.** A later audit of the ablation runs showed that the design measured more than grounding. Runs that sampled windows by predicted label mixed Normal and misclassified windows into the sample, so the score depended on the classifier as well as on the prompt. Only C3 carried a "use TR ..., e.g. TR 38.901" hint and KPI-specific field descriptions. The validator accepted any fault's primary standard for Jamming, which has no 3GPP reference, and the alignment table's move of Faulty Handover from TS 38.133 to TS 38.331 changed the score of earlier runs on re-scoring. The final design (Section 3.11; scoring rules in Appendix F) samples 3 windows per true fault type, scores each citation against the true fault's row and counts a missed window as incorrect. It gives all three conditions one instruction block and adds two answer rules that need no LLM as reference points.

**Final run.** Of the 30 windows sampled by true fault, the classifier labels 19 correctly, misclassifies 5 and misses 6. That leaves 24 windows with an explanation and 72 LLM calls, none of which fell back to the template (Section 5.10); the C3 evidence block was injected in all 24 explained windows. {tab:trackc} gives the counts, and {fig:trackc} shows the three LLM conditions next to the two rules without an LLM.

Table: In the final Track C run (30 windows, 3 per true fault type), C3 and the copy-row rule both cite a correct standard for 21 of 30 windows, against 5 of 30 for C1 and C2. "Correct" is scored against the true fault's row; "valid vs predicted" against the row of the fault the classifier named, over the 24 windows with an explanation. {#tab:trackc}

| Answer rule | Correct vs true fault (n = 30) | Wilson 95 % CI | Classified correctly (n = 19) | Valid vs predicted row (n = 24) |
| --- | --- | --- | --- | --- |
| C1 LLM: label + SHAP | 5/30 (16.7 %) | 7.3-33.6 % | 4/19 | 7/24 |
| C2 LLM: + past incidents | 5/30 (16.7 %) | 7.3-33.6 % | 4/19 | 7/24 |
| C3 LLM: + alignment row + RCA evidence | 21/30 (70.0 %) | 52.1-83.3 % | 19/19 | 24/24 |
| No LLM: always cite TS 38.141-1 | 12/30 (40.0 %) | — | — | — |
| No LLM: cite the predicted fault's row | 21/30 (70.0 %) | — | 19/19 | 24/24 |

![C3 reaches the 70 % threshold with 21 of 30 windows valid for the true fault and matches the rule that copies the predicted fault's row. C1 and C2 fall below the fixed answer TS 38.141-1 (final Track C run, three LLM conditions and two answer rules without an LLM).](figures/fig_trackc_citation.png){#fig:trackc}

Grounding raises citation correctness from 5/30 in C1 and C2 to 21/30 (70.0 %, Wilson 52.1 to 83.3 %) in C3, a gain of 53 pp. Moving from C2 to C3 improves 16 windows and worsens none (exact McNemar p = 3.1·10⁻⁵), and the same holds from C1 to C3 (16 better, 0 worse, p = 3.1·10⁻⁵). Retrieved incidents change nothing: C2 cites the same standard as C1 in all 24 explained windows. Without the table, the LLM cites TS 38.104 (12 windows), TS 38.331 (8) and TS 38.214 (4), and both C1 and C2 score below the fixed answer TS 38.141-1 (12/30). When the classifier's label is correct, which puts the correct standard in the prompt, C3 cites it in 19/19 windows (Wilson 83.2 to 100 %). In all 24 explained windows the LLM adopted the standard it was given (24/24 valid against the predicted row), so C3 scores exactly what the copy-row rule scores (21/30).

The run was repeated on 18 October 2026 after the team edited 61 of the 289 C3 evidence blocks to match the Layer 2 SHAP top three (Section 3.10). Only 3 of the 24 blocks that reach a C3 prompt changed (windows 607, 700 and 715), and C3 was identical to the previous run in every window. The C1 total did not change (window 658 now cites TS 38.104 instead of TS 36.104), and C2 fell from 7/30 to 5/30 (windows 684 and 686 now cite TS 38.331 instead of TS 38.321). The C1 and C2 prompts contain no RCA evidence, so these differences are run-to-run variation of the LLM at temperature 0.1, not an effect of the edit.

**Where the remaining errors come from.** {fig:trackc_out} breaks the 30 C3 outcomes down. In 19 windows the fault was classified correctly and the citation is correct. Two windows were misclassified but still score correctly: the CCI-Mild window 645 and the CCI-Severe window 703, both predicted as RF Filters, whose row lists TS 38.141-1 like their true rows. Three windows were misclassified and received the wrong row: 845 and 846 (Congestion-Gradual, predicted as Buffer Overflow) and 648 (Faulty Handover, predicted as RF Filters). The remaining six (717, 682, 879, 866, 827 and 843) were missed by the detector and have no explanation; they are all three Congestion-Sudden windows, one Congestion-Gradual, one Buffer Overflow and one Faulty Handover window. Every C3 error therefore starts upstream of the explanation layer.

![Under C3, 19 of the 30 Track C windows are classified correctly with a correct citation, 2 are misclassified onto a row with the same specification, 3 are misclassified onto the wrong row, and 6 are missed by the detector.](figures/fig_trackc_outcomes.png){#fig:trackc_out}

Window 611 (Doppler Shift, classified correctly; Appendix D) shows the difference in a single case. C1 and C2 both cite TS 38.104 and name the Near-RT RIC. C3 cites TR 38.901, names the O-RU / O-DU, and ties its root cause to the window's evidence: "Severe Doppler shift caused by high user equipment velocity resulting in rapid channel phase variation, evidenced by high RSRP variance (std |SHAP|=0.654, fft energy |SHAP|=0.510) and degraded downlink modulation (DL_MCS mean=6, |SHAP|=0.484)."

**From the first ablation to the final run.** {fig:trackc_evol} places the two runs side by side. The two bars of each pair measure different quantities. The first ablation counted a citation as valid against the predicted fault's row; the final run samples by true fault and counts a citation as correct only against the true row, with detector misses scored as incorrect. The fall from 27/30 to 21/30 in C3 is therefore a change of definition, not a decline. The final-run figure closest to the first ablation's definition is validity against the predicted row, 24/24.

![Each pair of bars sets the first ablation (valid against the predicted fault's row) beside the final run (sampled by true fault, correct against the true fault) for one condition. The two runs use different definitions, so a pair is not a like-for-like comparison.](figures/fig_trackc_evolution.png){#fig:trackc_evol}

**Interpretation.** The C3 figure of 21/30 measures how often the deployed system, from detection to explanation, cites a specification that governs the window's true fault; clause numbers are not checked. Because the correct standard is in the C3 prompt whenever the classifier is right, the 19/19 result measures how reliably the LLM adopts a standard supplied by the curated table and held to it by the validator. The LLM's own knowledge of 3GPP standards is measured by C1 (5/30), and the quality of the text around the citation by Track B (Section 5.8). The O4 criterion of 70 % was written for a "regex-validated 3GPP clause check"; the final run applies it at specification level.

### 5.8 Explanation quality: Track B

Track B, designed and run by Rodina Khallaf, scores KPI-RAG explanations of 30 fault windows, 3 per fault type across 10 fault types, with an LLM-as-judge protocol based on G-Eval [@liu2023geval] (Section 3.11). Each explanation is scored against a structured ground-truth reference on the four dimensions of {tab:app_geval} (Appendix F). No inference errors were reported across the 30 windows. {tab:trackb} and {fig:trackb} give the results per dimension. The composite mean is 3.59 / 5 with a standard deviation (SD) of 0.28. All four dimension means fall between 3.50 and 3.63, with actionability lowest (3.50), and every individual dimension score lies between 3 and 4.

Table: Actionability has the lowest Track B dimension mean (3.50), citation validity and fault specificity the highest (3.63), and every individual score is 3 or 4. Scores are on a 1 to 5 integer scale (n = 30 fault windows); the composite is the unweighted mean of the four dimensions. {#tab:trackb}

| Dimension | Mean (/ 5) | SD | Min | Max |
| --- | --- | --- | --- | --- |
| Citation validity | 3.63 | 0.49 | 3 | 4 |
| Fault specificity | 3.63 | 0.49 | 3 | 4 |
| Actionability | 3.50 | 0.51 | 3 | 4 |
| Causal soundness | 3.60 | 0.50 | 3 | 4 |
| Composite (mean of 4) | 3.59 | 0.28 | — | — |

![The four Track B dimension means lie within a narrow band of 3.50 to 3.63 on the 1 to 5 G-Eval rubric, and the composite is 3.59 (n = 30 windows).](figures/fig_trackb_dimensions.png){#fig:trackb}

On the citation dimension, the judge rated the cited specification correct, regardless of clause precision, in 19 of 30 windows (63.3 %); the other 36.7 % cite a valid TS document that is less directly applicable than the ground-truth reference. This rate comes from an LLM judge, so it is a different measure from the Track C counts of Section 5.7. Every window that cites the correct TS scores 4 and not 5 because it omits the specific clause (for example §7.8 of TS 38.141-1, or §5.4.3.1.3 of TS 38.321). The answer field of the explanation prompt asks only for a specification number ("TS XX.XXX or TR XX.XXX"). The Track B report proposes adding normative clause data to the knowledge base as the remedy.

{tab:trackb_fault} reports the dimension means per fault type (n = 3 windows each). For Doppler Shift, fault specificity is 4.00 because the mechanism is identified correctly, while actionability falls to 3.00 because the actions suggest generic channel-estimation improvements without parameter values. The Congestion-Sudden windows score 3.33 or below on every dimension. The judge notes that they are often conflated with coverage degradation, with RSRP and the signal-to-noise ratio (SNR) treated as causes instead of secondary effects, and that their burst-and-subside temporal signature is not captured in the KPI chain.

Table: Track B dimension means vary by fault type, and Congestion-Sudden scores 3.33 or below on every dimension (n = 3 windows per type, rows in the order of the Track B report). Cite = citation validity; Spec = fault specificity; Act = actionability; Causal = causal soundness. {#tab:trackb_fault}

| Fault type | n | Cite | Spec | Act | Causal |
| --- | --- | --- | --- | --- | --- |
| Antenna Failure | 3 | 4.00 | 3.67 | 4.00 | 3.67 |
| CCI-Mild | 3 | 4.00 | 3.67 | 3.67 | 3.67 |
| CCI-Severe | 3 | 4.00 | 3.67 | 3.00 | 4.00 |
| Faulty Handover | 3 | 3.67 | 4.00 | 4.00 | 3.33 |
| RF Filters | 3 | 3.67 | 3.67 | 4.00 | 3.67 |
| Resource Allocation Bugs | 3 | 4.00 | 3.33 | 4.00 | 3.33 |
| Doppler Shift | 3 | 3.00 | 4.00 | 3.00 | 4.00 |
| Congestion-Gradual | 3 | 3.33 | 3.67 | 3.00 | 3.67 |
| Buffer Overflow | 3 | 3.33 | 3.33 | 3.33 | 3.33 |
| Congestion-Sudden | 3 | 3.33 | 3.33 | 3.00 | 3.33 |

Across all 10 fault types, the judge credits the explanations for identifying the primary fault mechanism (for example, co-channel interference leading to an uplink SNR collapse and a throughput drop). It also credits them for naming the correct O-RAN component and for a directionally correct cause-to-KPI narrative. The judge records three recurring gaps. The first is the missing clause number described above. The second is an incomplete intermediate symptom chain: explanations usually skip the BLER rise and the floor in the modulation and coding scheme (MCS) that link the primary fault to the throughput KPI. Antenna Failure explanations, for example, go from a degraded antenna to low RSRP without the BLER rise that forces the lower MCS. The third is under-parameterised actionability: the actions identify the right subsystem and configuration area (MAC scheduler, physical cell identity (PCI) planning, buffer status report (BSR) timer) but give no parameter names, thresholds or target ranges. The Track B report identifies this gap as the largest single opportunity for improvement.

### 5.9 Knowledge scope: Track D

Track D, also by Rodina Khallaf, characterises the boundary of the system's knowledge outside fault diagnosis (Section 3.11). It compares the LLM without context and with context retrieved from the 935-ticket knowledge base on 600 TeleQnA questions [@maatouk2025teleqna], 200 from each of the Standards specifications, Standards overview and Lexicon categories. Track D therefore measures retrieval over that ticket corpus on out-of-domain questions.

At the coverage threshold τ = 0.50, only 13 of the 600 questions (2.17 %) are supported, and 193 (32.2 %) pass the lower τ = 0.35, the same value as the system's low-confidence flag. The median retrieval score across all 600 questions is below 0.31. The Track D report reads this as a correct self-assessment by the retriever: the 935 documents describe KPI anomaly events and contain almost no text close to standards specifications or glossaries. In the ungated run, however, the retrieved chunks are prepended to every question regardless of coverage. {tab:trackd} gives coverage and accuracy by category, and the left panel of {fig:trackd} compares the two conditions.

Table: On 600 TeleQnA questions, retrieval coverage at τ = 0.50 is 2.17 %, and ungated KPI-RAG retrieval lowers multiple-choice accuracy below the no-context baseline in every category, most in Lexicon. Δ = KPI-RAG minus baseline, in pp. {#tab:trackd}

| Category | Questions | Supported (≥ 0.50) | Coverage | Baseline | KPI-RAG | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Standards specifications | 200 | 10 | 5.0 % | 67.0 % | 61.0 % | −6.0 pp |
| Standards overview | 200 | 1 | 0.5 % | 75.0 % | 53.0 % | −22.0 pp |
| Lexicon | 200 | 2 | 1.0 % | 87.0 % | 41.5 % | −45.5 pp |
| Overall | 600 | 13 | 2.17 % | 76.3 % | 51.8 % | −24.5 pp |

Injecting the chunks lowers overall accuracy from 76.3 % to 51.8 % (−24.5 pp). The Track D report attributes the drop to context poisoning [@shi2023distracted]: conditioned on fault-diagnosis narratives about interference, jamming and congestion, the LLM partly overrides answers it would otherwise get right. One inspected case is question q00021, on the scrambling-sequence initialisation for the UE-specific search space of the physical downlink control channel (PDCCH), with the expected answer `pdcch-DMRS-ScramblingID` from TS 38.211. Its top chunk (score 0.217) described an external RF jamming incident, and the answer changed from correct to wrong. Lexicon loses most (−45.5 pp), because the baseline holds vocabulary definitions with high confidence, so irrelevant context is enough to displace them. Standards specifications loses least (−6.0 pp), because these questions are harder for the baseline as well and some chunks contain incidental technical terms.

![On 600 TeleQnA questions, ungated KPI-RAG loses accuracy against the baseline LLM in every category, most in Lexicon (left), and the overall accuracy of the gated system recovers to the 76.3 % baseline as the confidence threshold τ rises (right).](figures/fig_trackd.png){#fig:trackd}

To measure how much of the loss comes from forced injection, the Track D analysis applied a confidence gate after the fact to the stored answers, with no further LLM calls. The gate keeps the KPI-RAG answer only when the top retrieval score reaches τ and uses the baseline answer otherwise. The right panel of {fig:trackd} shows the sweep (full values in Appendix G). Accuracy rises steadily with τ, from 57.3 % at τ = 0.20 (496 questions use retrieval) to 69.7 % at τ = 0.35 (193 questions) and 75.2 % at τ = 0.45 (33 questions). At τ = 0.50, where exactly the 13 supported questions use retrieval, accuracy is 76.0 % (−0.3 pp), which recovers 98.8 % of the gap (24.2 of 24.5 pp). At that threshold Standards specifications returns to its baseline (67.0 %), while Lexicon (86.5 % against 87.0 %) and Standards overview (74.5 % against 75.0 %) each remain 0.5 pp below, from the 2 and 1 questions that clear the gate. At τ = 0.55 only one question uses retrieval and accuracy is identical to the baseline (76.3 %). The Track D report interprets Track D as a characterisation of scope and not as a failure on the system's own task. It proposes the confidence gate, which needs no further LLM calls, as a mitigation for RAG systems used outside their evidence domain.

### 5.10 Reliability and testing

The first prototype showed that the free-tier LLM quota was a practical constraint. Fixed linear backoff (30, 60 and 90 s) proved insufficient for evaluation-scale runs of 90 sequential calls, and a rate limiter was added (Section 4.5). The reported first ablation ended with two template fallbacks: a parse failure in C2 and three consecutive rate-limit (HTTP status 429) responses in C3. All 118 tests of the first prototype passed, including four negative tests that the first prototype's audit added to keep citation validation fault-specific (Section 3.10).

Under the final error-handling policy (Section 4.5), all 72 calls of the final Track C run returned a valid JSON answer with no template fallback, and four rate-limit responses were retried successfully. The Phase 1 report had set a target of fewer than 15 % template fallbacks; the final run had 0 of 72. The test suite, built by Ahmed Al Shobaki, grew to 210 tests in 29 modules, of which 207 pass on the final code and 3 are skipped (Section 4.7). Chapter 6 interprets these results against the research questions and the objectives.
