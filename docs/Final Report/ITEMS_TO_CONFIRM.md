# Items to confirm: KPI-RAG final report (Group 15)

This list collects every point in the merged final report (`KPI_RAG_Final_Report.md`) that one of us has to confirm, correct or decide before submission on 20 October 2026. Where the four source reports disagreed, the merged report currently uses each owner's own version for the owner's part and does not flag anything inside the text, so a wrong value will go out unless its owner answers here. Each owner replies to their items by ID (a short answer per item is enough, "OK as is" included), and the editor updates the report; please do not edit the report files directly.

Severity: **HIGH** = a number in the report may be wrong or cannot be traced; **MEDIUM** = a fact, attribution or wording that the report states but its owner has not confirmed; **LOW** = a small gap or a style choice. IDs starting with C come from the editor's conflict check; IDs starting with D are new. Section numbers refer to the assembled report.

---

## 1. Decisions for the whole team

### C37 · MEDIUM · Title page details
- **Where:** title page.
- **Report now says:** "CISC 898 002 Master's Project S26"; "Final Report (Phase 2), Group 15"; students in the order Ahmed Mohamed Al Shobaki, Farida El Ghabary, Raneem Alnaghy, Rodina Khallaf, with IDs (same order as the Phase 1 cover); "Department: School of Computing, M.S. / Professional Master's in Computer Science (AI & Data Science)"; "University: Queen's University, Kingston, Ontario, Canada"; "Industry partner: Capgemini"; "Date submitted: 20 October 2026".
- **Question:** "School of Computing" and "Kingston, Ontario, Canada" appear in no team document. Ahmed's individual report uses "... (AI & Data Science), DiGiLiANS Master Program, Round 1" and "Queen's University, in partnership with the National Telecommunication Institute (NTI)".
- **Answer needed:** (a) the exact department and programme line; (b) the university line, with or without the NTI partnership; (c) keep "Final Report (Phase 2)" (the course email calls this deliverable the "Phase-2 document") or drop "(Phase 2)"; (d) member order; (e) the submission date.

### C50 · MEDIUM · Is O4 "Met"?
- **Where:** 1.3 (O4), 3.11, 5.7 (Interpretation), 6.2 (Table 26), 7.1.
- **Report now says:** O4 is "Met" at the level the validator checks: 21/30 (70.0 %, Wilson 52.1-83.3 %) correct for the true fault, exactly the 70 % threshold. The text states that the proposal wrote O4 for a "regex-validated 3GPP clause check" and that the final check covers the specification number only.
- **Answer needed:** keep "Met (specification level)" or change to "Partly met". Ahmed: confirm the specification-level wording is accurate.

### D01 · MEDIUM · Status of O1, O2, O3 and O5
- **Where:** 6.2 (Table 26), 7.1.
- **Report now says:** O1 Partly met (the scale contribution was measured only under the superseded window-level split); O2 Met (rests on Rodina's "Confirmed" marks, whose verification method is not described, see C78); O3 Partly met (precision@k was measured against the true fault type, not on "held-out alignment-table queries", the proposal's criterion, which no report measured); O5 Met.
- **Answer needed:** agree, or give the status you want and the evidence for it. Farida and Raneem for O1, Rodina for O2, Ahmed for O3.

### C32 · MEDIUM · Who did what
- **Where:** 1.5, 3.1, 4.1, Appendix B (intro), Appendix K (Table 32).
- **Report now says:** Raneem: session reconstruction, session split, reject option and its promotion rule. Farida: v3 synthetic data, Model 2 recipe, "Data and models report (14 October 2026)". "ML team (Farida El Ghabary, Raneem Alnaghy)": frozen protocol, steps 01 to 12c, final Layer 2 pipeline. Rodina: alignment table v1.2 and causal mechanisms, Track B, Track D (inferred from the file name; her report has no byline). Ahmed "(team leader)": retrieval, prompt and evidence block, LLM layer and citation validation, Track C, retrieval experiment, evaluation scripts, dashboard, deployment, tests, packaging. Whole team: the first prototype.
- **Question:** each row is the editor's reading of the source reports; "team leader" comes only from Ahmed's own title page.
- **Answer needed:** each member corrects their own row. Farida and Raneem: who did which of steps 01 to 12c. Team: keep or drop "(team leader)".

### C14 · MEDIUM · Statements about the Phase 1 work
- **Where:** 3.3, 3.4 (Table 4), 5.1, 6.1, 6.3.
- **Report now says:** the first prototype's feature configuration "had been chosen on the test set"; the follow-up investigations "came from notebooks never executed in their submitted state"; the residual near-duplicate rate given for the session split "the later audit could not reproduce". The 4.05 % figure and the Phase 1 Table 6 values are not printed.
- **Question:** these findings come from the ML team's audit and describe work all four of us submitted in Phase 1.
- **Answer needed:** Farida and Raneem confirm the three findings; everyone agrees they can be printed.

### C23, C26, C27, C28, C51 · LOW · Vocabulary choices already applied
- **Where:** whole report.
- **Report now says:** "alignment table (matrix v1.2)" at first mention in each chapter, then "alignment table"; one short name per class (Table 3: CCI-Mild, CCI-Severe, Congestion-Gradual, Congestion-Sudden, RF Filters, Doppler Shift, Faulty Handover, ...), with Rodina's names kept inside Table 9 and in quotations; Track A detection and classification, Track B explanation quality (G-Eval), Track C citation ablation, Track D TeleQnA (3.11 notes that the Phase 1 plan had TeleQnA under Track B); "the first prototype (September 2026)" and "the final system (October 2026)" in place of Phase 1 and Phase 2 ("Phase 1 report" only when naming that document); "injected faults" for the 10 TelecomTS fault types and "synthetic training windows (v3)" for Farida's data; "KPI narratives" for the retrieval documents, "tickets" for the dataset field, "RCA evidence records" for `rca_evidence.json`; the proposal's O1-O5 and RQ1-RQ3 (Ahmed's individual RQs are not used); British spelling.
- **Answer needed:** reply only if you object to a choice.

### D02 · LOW · Abstract length
- **Where:** Abstract.
- **Report now says:** one paragraph of about 365 words. The course asks for "a brief summary ... in a paragraph" and gives no limit.
- **Answer needed:** OK, or a word limit to cut to.

### D03 · LOW · Acknowledgements
- **Where:** Acknowledgements.
- **Report now says:** thanks Prof. Yuanzhu Chen, Capgemini, Queen's University (programme and the Frontenac cluster), the NTI and the Ministry of Communications and Information Technology of Egypt "which run the DiGiLiANS Master Program", and the TelecomTS and GSMA knowledge-graph authors. The NTI, Ministry and DiGiLiANS sentence comes from Ahmed's individual report.
- **Answer needed:** confirm the sentence applies to all four members; name anyone else to thank.

### D04 · LOW · Novelty claims
- **Where:** 1.2, 2.6; claims dropped from 1.5 and 7.2.
- **Report now says:** "To the best of our knowledge, no published system addresses all three sub-problems in a single evaluable pipeline on public benchmark data" (1.2) and a similar sentence in 2.6. Two Phase 1 claims were dropped: the "first per-class F1 table for the 11-class TelecomTS task" (no 11-class per-class table exists in any report) and the "first ... alignment mapping with independently verified evidence spans".
- **Answer needed:** keep the two hedged claims as worded; say if a "first" claim should come back (the per-class one needs Farida's full 11-class table).

---

### D25 · MEDIUM · Source of the Mantis F1 0.800 (owner: whole team, Farida El Ghabary for the ML history)
- **Report now says:** the binary-detection F1 of 0.800 for Mantis is a result of the TelecomTS paper's own benchmark (its Table 2), cited to the TelecomTS paper; Toto's 0.848 is a root-cause accuracy from its Table 4. Both come from a benchmark that uses a random split of overlapping windows, so neither is compared directly with the session-level results.
- **Why it changed:** the Phase 1 report credited 0.800 to the Mantis paper (arXiv 2502.15637, February 2025), which predates TelecomTS (arXiv 2510.06063, October 2025) and cannot contain a TelecomTS result. The proposal already gave the correct source ("Mantis (F1=0.800, TelecomTS Table 2)").
- **Answer needed:** confirm the correction.

## 2. Farida El Ghabary

### C09 · MEDIUM · What "reproduced to four decimals" refers to
- **Where:** 3.4, 5.3 (Synthetic data), Appendix B step 01.
- **Report now says:** "the ML team reproduced the real-only baseline (0.2908 ± 0.0902) and her final v3 model (0.3743, in 25 of 25 folds) to four decimals". The Phase 1 report's 0.333 ± 0.076 appears only as a five-fold estimate "from a different protocol"; no sentence compares 0.333 with 0.389.
- **Question:** your summary table placed 0.291 under "Phase 1 (12-Sep)", but no Phase 1 value is 0.2908. What exactly was reproduced?
- **Answer needed:** the runs that 0.2908 and 0.3743 reproduce, or permission to drop "to four decimals".

### C18 · MEDIUM · What the 496 features are (with C66)
- **Where:** 3.5 (Table 5 and text), 4.2, 4.3 (Table 8), 5.1 (Table 13), Appendix B steps 05a and 07.
- **Report now says:** "496 features (Config B + FFT + G1 G2 G3)"; 3.5 calls them "Farida El Ghabary's recipe (Config B and FFT features) with the trend and co-movement features"; "CatBoost (240 features)"; step 05a appears only as "the level-free result". Config B is defined only in its Phase 1 sense (320 dimensions).
- **Answer needed:** one line each: is your Config B the Phase 1 320-dimension Config B; what FFT and G1, G2, G3 contain (are G1-G3 the trend and co-movement features); what the 240-feature set is; what step 05a did.

### C59 · MEDIUM · The other 7 false alarms
- **Where:** 5.4 (Remaining errors).
- **Report now says:** "116 false alarms, including 75 labelled CCI-Mild, 20 Faulty Handover, 8 Antenna Failure and 6 Doppler Shift". These sum to 109.
- **Answer needed:** the classes of the remaining 7 (the report then drops "including").

### C64 · MEDIUM · How 4,250 synthetic windows arise
- **Where:** 3.6, 4.3 (Table 8), 5.3 (Table 16), Appendix B step 09a.
- **Report now says:** Model 2 is trained on "935 real and 4,250 synthetic (v3) windows (5,185 rows)", and the volume test decided "Keep 500" per class. 500 per class gives 5,500 for 11 classes or 5,000 for 10.
- **Answer needed:** the per-class counts or rule that gives 4,250.

### D05 · MEDIUM · Why the final Model 2 was chosen
- **Where:** 3.6, 5.3 (Table 17 and text), Appendix B step 09c.
- **Report now says:** the final model is the recipe plus trend features (CV 0.389; held-out macro-F1 0.591, accuracy 0.703). Table 17 shows higher CV scores for QUANT (0.426), CatBoost + catch22 (0.393) and both soft votes (0.453, 0.446), and CatBoost + catch22 also has better held-out scores (0.608 / 0.730). The text only says catch22 "is only 0.004 above the final model in cross-validation, and the held-out sessions were never used for selection".
- **Answer needed:** one or two sentences on the rule that selected the recipe plus trend features over these candidates.

### D06 · MEDIUM · Definition of "useful precision"
- **Where:** 5.4 (Table 18).
- **Report now says:** a column "Useful precision" with 0.521, 0.549 and 0.545 and no definition.
- **Answer needed:** the definition (formula) for the caption.

### D07 · MEDIUM · Answers to RQ1 and RQ3 (with Raneem)
- **Where:** 6.1, 5.1 (last paragraph), 1.3 ("kept throughout").
- **Report now says:** both are "answered only in part"; "None of the source reports repeats the three-configuration ablation under the session-level protocol"; "No source reports training or inference cost next to the feature configurations, so the trade-off can be stated in dimensionality only."
- **Answer needed:** confirm, or send any session-level ablation or cost numbers.

### C13 · LOW · Mantis and Toto (resolved by the proposal; confirm only)
- **Where:** 2.2, 5.5 (Table 20).
- **Report now says:** both figures come from the TelecomTS paper's own benchmark, which uses a random split of overlapping windows (your statement): Mantis F1 0.800 for binary detection (TelecomTS Table 2) and Toto accuracy 0.848 for fault-type classification (TelecomTS Table 4), as the proposal states. Neither is compared directly with the session-level results. See also D25.
- **Answer needed:** none unless this reading is wrong.

### C20 · LOW · SHAP fields in the Layer 2 JSON
- **Where:** 3.8, 4.1.
- **Report now says:** "The value is the magnitude |SHAP|, with the sign kept in another field"; each entry also stores its effect on the predicted class and the KPI's direction against its normal mean (`feature_vs_normal`).
- **Answer needed:** confirm these field meanings.

### C21 · LOW · Calibration wording
- **Where:** 3.6, 4.3, 5.4 (Calibration).
- **Report now says:** "A teammate reported that it erased the ranking information, the ML team confirmed it, and the calibrator was removed"; confidence is raw "as in the first prototype". Your sentence that Phase 1 "labelled it uncalibrated" is not repeated (the Phase 1 report does not use that word).
- **Answer needed:** name the teammate if you want them named; point to the Phase 1 source if the "uncalibrated" remark should return.

### C61 · LOW · Same models, two CV scores
- **Where:** 5.1 (Table 13), 5.3 (Table 17), Appendix B steps 07 and 09c.
- **Report now says:** QUANT 0.428 (step 07) and 0.426 (step 09c); CatBoost + catch22 0.402 and 0.393. 5.3 says the later values "come from this later step and differ slightly".
- **Answer needed:** why they differ (one phrase), or OK as is.

### C63 · LOW · How the CV F1 is averaged
- **Where:** 5.2 (Tables 14 and 15).
- **Report now says:** the adopted detector has CV precision 0.440 and recall 0.499 (Table 14) and CV F1 0.376 (Table 15). Pooled, 0.440 and 0.499 would give about 0.468.
- **Answer needed:** whether 0.376 is a mean of per-fold F1 (added to the caption).

### C65 · LOW · Which gain the bootstrap interval belongs to
- **Where:** 5.4 (Uncertainty), Appendix B step 12c.
- **Report now says:** +0.098 macro-F1, 95 % CI [+0.078, +0.117], Cohen's d = 1.9, for the final model over the real-only baseline only. Your text said "the gains of synthetic data and of the final model".
- **Answer needed:** confirm.

### C84 · LOW · Phase 1 detector comparison printed as history
- **Where:** 5.1 (Table 13, upper part).
- **Report now says:** Random Forest 0.665, XGBoost 0.533, LightGBM 0.364, CatBoost 0.802 (binary F1, Phase 1 Table 7, marked superseded), with a note that the Phase 1 report does not reconcile 0.802 with its natural-rate F1 of 0.321.
- **Answer needed:** keep or drop the upper part of Table 13.

### D08 · LOW · Three readings in 5.2
- **Where:** 5.2 ("Why every detector fails on session 0", "Targeted alternatives").
- **Report now says:** (a) "In cross-validation, 90 % of the remaining false alarms come from two Normal-only training sessions, 0 and 5" (your text has no "in cross-validation"); (b) session 0's "Normal windows score about 0.99 and its real Jamming windows about 0.01", read as Model 1's anomaly score; (c) "the scale ablation of the TelecomTS paper [1]" for your "the paper's scale ablation".
- **Answer needed:** confirm or correct each.

### D09 · LOW · The adopted detector is not the best CV row
- **Where:** 5.2 (Table 14).
- **Report now says:** "Fix weighting" PR-AUC 0.498, "Fix both (adopted)" 0.492; "The variant with both fixes was adopted." No reason is given.
- **Answer needed:** the reason (for example, it removes both defects), or OK as is.

### D10 · LOW · The ninth algorithm
- **Where:** 3.6, 5.1 (Table 13).
- **Report now says:** "Nine algorithms were compared ... the lower part of Table 13 lists the eight for which scores are reported."
- **Answer needed:** name and score of the ninth (the baseline CatBoost?).

### D11 · LOW · Parser fix in the generator
- **Where:** 3.5, 5.3 (Table 16), Appendix B.
- **Report now says:** fixing the parser in the generator "gave a cross-validated macro-F1 of 0.385, a difference the ML team judged negligible"; the value it was compared with is not stated.
- **Answer needed:** the comparison value (0.389?).

### D12 · LOW · Description of the v3 method
- **Where:** 3.7, 4.2.
- **Report now says:** the generator "builds synthetic fault windows by copying fault effects from the real fault sessions onto Normal-session backgrounds ... Donors and hosts come only from the 28 training sessions".
- **Answer needed:** confirm this describes v3 correctly.

### D13 · LOW · Isolation Forest baseline
- **Where:** 5.2, 5.5 (Table 20).
- **Report now says:** Isolation Forest F1 0.227, evaluation set "Not stated" in the Phase 1 report.
- **Answer needed:** the evaluation set, if anyone remembers it.

### D14 · LOW · Tools line and run instructions
- **Where:** 3.12, Appendix B (last paragraph).
- **Report now says:** aeon is listed among the ML libraries without its use; Appendix B gives one command (`01_freeze_data_and_reproduce_baseline.py   # ... through 11_final_pipeline.py`).
- **Answer needed:** what aeon is used for (QUANT, MiniRocket, Hydra?); a full step list if you want one in Appendix B.

### D15 · LOW · "Pre-registered"
- **Where:** 5.1 ("under the pre-registered adoption rule"), Appendix B step 08 ("by the pre-registered rule"); 7.1 says "pre-stated adoption rule".
- **Report now says:** the rule was frozen in step 01; no external registration is cited.
- **Answer needed:** keep "pre-registered" or change to "frozen" / "pre-stated".

---

## 3. Raneem Alnaghy

Also for you: C32 (your contribution row), C14 (Phase 1 audit findings), D01 (O1 status), D07 (RQ1 and RQ3).

### C15 · MEDIUM · Sessions that contain anomalies
- **Where:** 3.3, 1.4, 6.5, 7.3.
- **Report now says:** "Under this reconstruction, 13 of the 33 sessions contain anomalies"; "The training faults therefore come from 10 sessions, two with Jamming." The earlier "2 of 33" is not used.
- **Question:** is 13 = 10 training sessions + 3 test sessions (1, 13, 17)?
- **Answer needed:** confirm 13 and the 10 + 3 split.

### D16 · LOW · Do the five seeds give different folds?
- **Where:** 3.4 (Table 4), Appendix B step 01.
- **Report now says:** "StratifiedGroupKFold by session, 5 folds × 5 seeds" (25 folds). The Phase 1 report said repeated five-fold CV with three seeds "produced identical fold partitions".
- **Answer needed:** confirm the five seeds give different partitions, or say how the 25 folds differ.

---

## 4. Rodina Khallaf

Also for you: C32 (your contribution row), D01 (O2 status), and the references in Section 6 (C35, D20).

### C01 · HIGH · Which explanations Track B scored (with C02, C25, C52)
- **Where:** 3.11, Table 7, 5.8, 6.5 (Provenance), Appendix F.
- **Report now says:** Track B scores "KPI-RAG explanations of 30 fault windows, 3 per fault type across 10 fault types"; "No inference errors were reported across the 30 windows"; the judge model is not named and "C3 dataset" is not used; the 63.3 % (19/30) is called "judge-rated citation validity", "a different measure from the Track C counts", and is not compared with them; 6.5 says the scores "cannot yet be tied to a particular explanation set".
- **Question:** the final Track C run explains only 24 of its 30 windows and none of the three Congestion-Sudden (Sudden Spike) windows, which the detector misses. Track B scores three Sudden Spike windows with no errors, so it scored an earlier set of explanations.
- **Answer needed:** the explanation file or run and its date; the generator LLM and prompt condition; the judge LLM; whether windows were sampled by true or predicted fault type (any Normal windows among them?); what "C3 dataset" means; whether "zero inference errors" refers to the judge or the generator; and the reference standard the judge counted as correct for each fault. Ahmed can help identify the run (an earlier Track C output from 16 October 2026, commit fe72712, had 3 windows per predicted type including Sudden Spike).

### C70 · HIGH · Track B per-type composites and score distribution (with C71, C72, C73)
- **Where:** 5.8 (Tables 23 and 24).
- **Report now says:** the four dimension columns per fault type, in your row order. Left out until recomputed: your composite column, the ranking (Antenna 3.87 highest, Sudden Spike 3.27 lowest), the score-distribution figure (5.Y.1), the sentence "clusters at 3.6-3.8 (16 of 30 windows)", the composite min and max, and the sentence about no window reaching a perfect 4.0.
- **Question:** each composite should be the mean of its four columns but is not (Antenna 3.87 against 3.83, Doppler 3.40 against 3.50, Sudden Spike 3.27 against 3.25, and seven more). With integer scores and 3 windows a type composite must be a multiple of 1/12, so 3.87, 3.80, 3.73, 3.40 and 3.27 cannot occur. A single window's composite can only be 3.00, 3.25, 3.50, 3.75 or 4.00, so the figure's bins (3.4, 3.6, 3.73, 3.8, 3.87) cannot occur either. The composite maximum of 4.0 contradicts "no window achieving a perfect 4.0 composite".
- **Answer needed:** the per-window scores, or recomputed per-type composites, distribution and min/max; and confirmation that the per-dimension columns themselves are right.

### C05 · HIGH · Track D setup (with C76)
- **Where:** 1.5, 3.11, Table 7, 5.9, 6.5, Appendix G.
- **Report now says:** "the retriever queries a ChromaDB knowledge base of 935 fault-diagnosis telemetry tickets and prepends the top-k chunks"; "Track D therefore evaluates retrieval over this ticket corpus, not the deployed retrieval over per-window narratives"; the LLM, embedding model and k are not named (listed as a limitation in 6.5); "pre-registered" and "two-stage" are dropped; the design is two conditions plus a post-hoc gate.
- **Question:** your example chunk ("Issue: Severe communication degradation ...") is ticket text, while the deployed index holds per-window KPI narratives.
- **Answer needed:** which index was queried (ticket index or the deployed `telecomts_descriptions` narrative index) and when; the LLM; the embedding model; k; a reference if the protocol was registered anywhere. Ahmed: whether the ticket index still existed at that date.

### C03 · MEDIUM · Antenna Failure standard
- **Where:** Table 9 (row #1), Appendix A, 5.8 (Table 24).
- **Report now says:** TS 38.141-1 for Antenna Failure everywhere (team decision, matching your table and Track C). Your sentence "the ground-truth TS (38.104) is reliably cited" is not reproduced. Antenna's Track B citation score (4.00) is printed.
- **Answer needed:** confirm "38.104" was a slip and that the judge's reference for Antenna Failure was TS 38.141-1. If it was TS 38.104, the Antenna citation scores need rescoring.

### C19 · MEDIUM · Source of the "Affected KPIs (SHAP Top 3)" column
- **Where:** 3.9, 4.4 (Table 9).
- **Report now says:** the column "reflects Layer 2 mean absolute SHAP contribution rankings across evaluated time windows" (your wording); it is not described as the final Model 2's ranking.
- **Answer needed:** which model and JSON produced it (Phase 1 64-feature model or final 496-feature model) and over which windows. Farida: whether it matches the final Layer 2 output.

### C24 · MEDIUM · Alignment table versions
- **Where:** 3.9, 3.10, 4.4.
- **Report now says:** the Phase 1 table is called "the first table", not v1.0 or v1.1; 4.4 says "some RCA evidence records still carry the older v1.1 standard (Handover TS 38.133)" (Ahmed's wording); the Handover move from TS 38.133 §4.3.5 to TS 38.331 §5.5.4.4 and §5.5.4.1 is stated without a reason.
- **Answer needed:** what v1.0, v1.1 and v1.2 were and when; whether the Phase 1 table was v1.0 or v1.1; optionally one sentence on why Handover moved to TS 38.331.

### C31 · MEDIUM · Who built the RCA evidence records
- **Where:** 3.10, 4.5, 5.7, Table 6.
- **Report now says:** "289 records, one per non-Jamming alarm", described by content only; "Before the final Track C run the team revised the records to match the Layer 2 SHAP top three; the revision changed the C3 evidence block of 61 of the 289 records." No builder is named and no section says how they were built. Your report's title mentions an "RCA Evidence Engine" but has no section on it.
- **Answer needed:** who built `rca_evidence.json` and how (one or two sentences for 3.10 and Appendix K). Farida, Raneem or Ahmed answer instead if it was theirs, and whoever made the 18 October edit says whether to be named.

### C74 · MEDIUM · CCI-Mild and RSRP
- **Where:** Appendix A (Group 3) against Table 9 (row #3).
- **Report now says:** Appendix A: "mild interference (#3) causes a moderate `UL_SNR` drop while RSRP remains unchanged"; Table 9 lists `UL_SNR`, `RSRP`, `UL_Pkts` for CCI (Mild).
- **Answer needed:** reword the sentence or change the row.

### C78 · MEDIUM · How v1.2 was verified, and clause details
- **Where:** 3.9, 4.4, 6.2 (O2), Appendix A.
- **Report now says:** "its author reports the 10 injected classes as verified against Release 19", with no method. Editorial changes made to your text: "ETSI TR 38.901" to "3GPP TR 38.901"; "UDM" removed from "ping-pong handovers"; "(Dual Mobility)" removed after TR 38.901 §7.6.10.
- **Answer needed:** the verification method in one or two sentences (the O2 "Met" rests on it); the title of TR 38.901 §7.6.10; confirm the three edits.

### C22 · LOW · Meaning of "Dataset Validated"
- **Where:** 3.9, 4.4, Table 9 (row #11), Appendix A (Jamming).
- **Report now says:** Jamming is marked "Dataset Validated": "a real over-the-air condition, evaluated against operational ticket logs, with no 3GPP clause or O-RAN target".
- **Answer needed:** what "Dataset Validated" means and which ticket logs were used.

### C75 · LOW · KPI names in Table 9
- **Where:** 4.4 (Table 9).
- **Report now says:** your names verbatim (`PRB_Util_UL`, `UL_Pkts`, `PRBs_UL_Curr`, `PRBs_DL_Curr`); the prose and Appendix A use the dataset names (`PRB_Utilization_UL`, `PRBs_UL_Current`, ...).
- **Answer needed:** switch the table to the dataset channel names, or keep.

### D17 · LOW · British spelling of 3GPP names
- **Where:** Appendix A (Group 2).
- **Report now says:** "Logical Channel Prioritisation" and "Prioritised Bit Rate"; 3GPP spells them with z.
- **Answer needed:** keep, or use the 3GPP spelling for these names.

### D18 · LOW · Rubric level descriptors
- **Where:** Appendix F (Table 29).
- **Report now says:** the four dimensions with the judge's question for each and "a five-point rubric calibrated against expert-written ground-truth explanations"; no description of what a 1, 3 or 5 means.
- **Answer needed:** the level descriptors if Appendix F should show them; who wrote the expert explanations.

---

## 5. Ahmed Al Shobaki

Also for you: C01 and C05 (identify the runs and the index), C31 (the 18 October edit), C50 (O4 wording), D01 (O3 status), C32 ("team leader"), D23 (references).

### C07 · MEDIUM · Sampling of the first ablation and "the first run"
- **Where:** 3.11 (Track C), 5.7 ("Why the protocol changed"), 6.3.
- **Report now says:** 3.11: "The first prototype drew 3 test windows for each fault with a 3GPP reference (seed 42) and scored validity against the predicted fault's row; a later audit of the ablation runs found that sampling windows by predicted label mixed Normal and misclassified windows into the sample." 5.7: "Runs that sampled windows by predicted label ..." and the Handover change "changed the score of earlier runs on re-scoring". Your counts "8 of its 30 windows were normal traffic, 9 were misclassified" and "30/30 into 28/30" are not printed.
- **Question:** the Phase 1 run had C3 = 27/30 with one fallback, so it cannot be the 30/30 run, and the Phase 1 report does not say whether it sampled by true or predicted type.
- **Answer needed:** did the Phase 1 run (27/30) sample by predicted label? Which run had the 8 Normal windows, and which went from 30/30 to 28/30? Or confirm the current general wording is enough.

### C08 · MEDIUM · The early L2 index
- **Where:** 3.10, 4.5 (Table 11).
- **Report now says:** "an early index had used L2 distance while computing similarity as one minus the distance", without saying which index. The Phase 1 report says its index was in "cosine distance space".
- **Answer needed:** was the L2 index the Phase 1 index (then the Phase 1 statement was wrong) or a later one?

### C33 · MEDIUM · Role wording and test-suite scope
- **Where:** 1.5, 3.1, 4.7, 5.10, Appendix J, Appendix K.
- **Report now says:** your role includes "the Track C evaluation and evaluation tooling" (3.1); "The test suite, maintained by Ahmed Al Shobaki" (5.10, Appendix J); the suite includes `test_phase2_data`, `test_phase2_models`, `test_phase2_pipeline` (53 tests) and `test_run_eval_track_b` (4 tests), and the report does not say whose code these cover.
- **Answer needed:** who wrote the `test_phase2_*` and Track B test modules, and whether "maintained by Ahmed Al Shobaki" holds for the whole suite (Farida to agree for the ML modules).

### D19 · LOW · GALA row in Table 1
- **Where:** 2.4 (Table 1).
- **Report now says:** GALA's domain is "root-cause analysis with graph information" (editor's wording; your Table 7 omits GALA).
- **Answer needed:** the correct domain and input.

---

## 6. References to confirm

### C35 · MEDIUM · Rodina's citations (owner: Rodina)
- **Gao et al., 2023.** The report no longer cites it: the sentence saying G-Eval correlates strongly with expert judgements was removed because the reference could not be identified from "[Gao et al., 2023]" alone. Send the full reference if the sentence should be restored.
- **Shi et al., 2023 [34].** The editor completed it as F. Shi *et al.*, "Large language models can be easily distracted by irrelevant context," ICML 2023 (cited in 2.3 and 5.9). Confirm.
- **Lewis et al., 2020.** Your report cites it with Shi for "context poisoning". The merged report cites Lewis only for retrieval-augmented generation in general and Shi alone for the distraction effect. Confirm this is acceptable.
- **TeleQnA [33].** You cite "Maatouk et al., 2023"; the report uses the *IEEE Network*, 2025 version. Confirm.
- **O-RAN documents.** The report cites only [4] O-RAN.WG1.OAD-R003-v07.00 (2022, from Phase 1). Your print note lists O-RAN.WG4.MP, O-RAN.WG6.O2 and O-RAN.WG3.E2, which your text never cites. If you used them for the O-RAN targets in Table 9, send their versions.

### D20 · MEDIUM · Release and version of the 3GPP specifications (owner: Rodina; whole team for TS 38.133)
- **Report now says:** [10] TS 38.141-1, [11] TS 38.104, [12] TS 38.321, [13] TS 38.331, [14] TS 28.552 and [15] TR 38.901 are given as "Release 19" with no version number or date; [46] TS 38.133 (the Phase 1 Handover row, history only) and [50] TS 38.211 (Track D example) have no release at all. The Phase 1 report dated its 3GPP entries 2022.
- **Answer needed:** the exact version (for example V19.x.0) and date of each document used, so that the clause numbers in Table 9 can be traced; the release of TS 38.133 as used in Phase 1 and of TS 38.211.

### D21 · LOW · References added by the editor for methods in the ML report (owner: Farida)
- **Report now says:** your report names these methods without references, so the editor added [22] ExtraTrees (Geurts *et al.*, 2006), [23] Isolation Forest (Liu *et al.*, 2008), [24] QUANT (Dempster *et al.*, 2024), [25] MiniRocket (2021), [26] Hydra (2023), [27] catch22 (Lubba *et al.*, 2019), [43] Nadeau and Bengio (2003) for the corrected t-test, and [44] Zadrozny and Elkan (2002) for isotonic calibration.
- **Answer needed:** confirm these are the methods you used; add a library citation (for example aeon) if wanted.

### D22 · LOW · Toto (owner: Farida)
- **Report now says:** Toto has no reference entry; "0.848 for the Toto model" is cited to the TelecomTS paper [1] (2.2, 3.11, 5.5).
- **Answer needed:** confirm the number appears in [1] (and where), or give its source.

### D23 · LOW · Online references (owner: Ahmed)
- **Report now says:** [7] Gemini API documentation, [9] Streamlit documentation and [31] Chroma have a URL but no access date or version.
- **Answer needed:** access dates (and versions, if wanted) for IEEE style.

### D24 · LOW · Knowledge-graph entries (owner: whole team, from Phase 1)
- **Report now says:** [2] "Y. Yang *et al.*, telecom-kg-rel19: A 3GPP Release 19 knowledge graph, Hugging Face, 2026" and [3] "GSMA, O-RAN specification knowledge graph, Hugging Face, 2026", both without a URL.
- **Answer needed:** the URLs and the correct authors.

---

## 7. Resolved during editing (no answer needed)

Merged into other items: C02, C25 and C52 into C01; C71, C72 and C73 into C70; C76 into C05; C66 into C18. Track C values follow the final run (C1 5/30, C2 5/30, C3 21/30, +53 points); the older C2 = 7/30 and +47 points no longer appear.

- C04: Rodina's clause-omission finding and the prompt's answer field are given side by side (5.8); no cause is asserted.
- C06: tickets described with both facts, 1,235 generated texts and 11 distinct texts in training (Table 2, 3.10).
- C10: Phase 1 0.677 and 0.647 printed with Phase 1 definitions (Table 19); "≈ 203" and any direction of change left out.
- C11: 0.537 attributed to Farida's recipe, 0.553 to the Phase 1 single split (5.3).
- C12: Phase 1 baseline is F1 0.32 at the natural rate; 0.698 shown only as a rebalanced-subset figure (5.2, Table 20).
- C16: "consistent, not significant" attributed to Farida's own post-prototype evaluation (3.7, 5.3).
- C17: 18 channels, 16 numerical KPIs and 2 categorical fields; models and SHAP use the 16 KPIs (2.1, 3.2).
- C29: TeleQnA literature claim kept with its condition and linked to Track D only through Rodina's reading (2.3, 6.1).
- C30: dashboard runs retrieval and the explanation live on stored or uploaded classifier outputs (1.4, 4.6).
- C34: "first per-class F1 table" not claimed (see D04).
- C36: TelecomTS references [1] and [6] consistent.
- C38: no Jamming-alarm count printed; 289 records stated as given.
- C39: final session split from Raneem's work; the window-level split appears only as history (3.3).
- C40: final feature sets per the ML report; Model 2 never described as 64 features (Tables 5 and 8).
- C41: final Model 1 values from the ML report (5.2).
- C42: final Model 2 values, each with its denominator (5.3, 5.4).
- C43: catch-all share 82.2 % to 64.7 % (75 of 116) (5.4, Table 19).
- C44: the reject option's 83 % shown only as the reason it looked promising (5.4).
- C45: defects told in order in 3.5, 5.2 and 6.3.
- C46: alignment row history summarised in 3.9 and 4.4.
- C47: retrieval history and final design in 3.10 and 5.6.
- C48: RCA evidence before and after in 3.10 and 4.5 (builder still open in C31).
- C49: Track C final values taken from the final run; intermediate runs not mentioned (5.7).
- C53: Track D plan and execution described in 3.11 and 5.9.
- C54: tests 118 then 209 (206 pass, 3 skipped) (4.7, 5.10).
- C55: dashboard panels and inputs from Ahmed's report (4.6).
- C56: LLM backend and retry policy as in Ahmed's report (3.10, Table 10).
- C57: Jamming validator history in Ahmed's wording (3.10, 5.7).
- C58: SHAP source and SHAP-in-prompt changes in 3.8 and 4.3.
- C60: placeholder replaced by 64.7 % (75 of 116).
- C62: every accuracy printed with its denominator.
- C67: argmax rejection (about 20 %) and 13 % at τ = 0.5 printed with their rules (5.4).
- C68: formatting artefacts of the ML report not carried over.
- C69: precision written "0.70 (266 of 382 alarms)"; "10 training sessions that contain faults".
- C77: 2.17 % in the text, 2.2 % only in the rounded sweep column (Appendix G); τ = 0.55 "identical to the baseline".
- C79: Rodina's sections renumbered; the unquantified "prevalence" clause dropped ("zero inference errors" is part of C01).
- C80: fallback models not mentioned; the model-name caveat kept (6.5).
- C81: gain stated as +53 points from C1 and from C2 to C3.
- C82: only the first ablation and the final run are named.
- C83: Wilson intervals and McNemar tests described as descriptive (3.11, 5.7).
- C85: order of the split fix and the algorithm selection not asserted (5.1).
- C86: "582 to 584"; no 584-dimension result claimed.
- C87: Phase 1 Table 6 value 0.984 not printed; Table 5 values shown only as superseded history (Table 12).
- C88: "26:1" replaced by counts.
- C89: precision 0.196 used, not 0.209.
- C90: C3 composition taken from the Phase 1 text, not its figure label.
- C91: unexecuted Phase 1 plans not reported as results.
- C92: only the 0.35 retrieval threshold used.
- C93: Phase 1 wording slips not copied.
- C94: the winner of the window-split algorithm series not named.
