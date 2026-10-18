::: {.title-page}
[Master's Project Report]{.kicker}

CISC 898 002 Master's Project S26

# KPI-RAG: Explainable Root-Cause Analysis for 5G Networks via Standards-Grounded Retrieval-Augmented Generation

**Final Report (Phase 2), Group 15**

**Students:** Ahmed Mohamed Al Shobaki (ID: 20596291), Farida El Ghabary (ID: 20595451), Raneem Alnaghy (ID: 20596302), Rodina Khallaf (ID: 20596324)

**Supervisor:** Prof. Yuanzhu Chen

**Department:** School of Computing, M.S. / Professional Master's in Computer Science (AI & Data Science)

**University:** Queen's University, Kingston, Ontario, Canada

**Industry partner:** Capgemini

**Date submitted:** 20 October 2026
:::

<!-- PAGEBREAK -->

## Abstract

Diagnosing a fault in a 5G radio access network means connecting anomalous key performance indicators (KPIs) to a physical cause, the responsible network component and the governing specification of the 3rd Generation Partnership Project (3GPP), a task still largely done by hand. This project built and evaluated KPI-RAG, a pipeline that detects, classifies and explains faults on the public TelecomTS dataset (32,000 windows, 1,235 anomalous, 11 fault types), with explanations whose cited standards can be checked automatically. A two-stage CatBoost cascade with SHapley Additive exPlanations (SHAP) passes each alarm to a retrieval-augmented explanation layer. This layer gives a large language model (LLM) the predicted fault's row of a 3GPP Release 19 and O-RAN alignment table and validates the cited specification against that row. The first prototype (September 2026) established the pipeline and a split by 33 recording sessions. For the final system (October 2026), the team corrected two detector defects, added synthetic fault data and trend features to the classifier, and redesigned retrieval after finding that the dataset's tickets are one template per fault class. On five held-out sessions, false alarms fell from 1,105 to 116 (precision 0.70, recall 0.887, F1 0.78), and the classifier's accuracy was 0.703 (macro-F1 0.591). In an ablation on 30 test windows sampled by true fault type, the cited standard was correct for 5 of 30 windows without the alignment table and for 21 of 30 (70.0 %) with it. With the table it was correct for all 19 windows the classifier had labelled correctly, and every remaining error started in detection or classification. Separately, an LLM judge scored 30 explanations at 3.59 out of 5. On the TeleQnA telecom benchmark, a retrieval confidence gate removed almost all of the accuracy lost to irrelevant context (76.0 % against a 76.3 % baseline). Standards grounding is therefore dependable once the fault is identified, and further gains depend mainly on detection and on more fault recordings.

**Keywords:** 5G, O-RAN, root-cause analysis, anomaly detection, retrieval-augmented generation, 3GPP, SHAP, CatBoost, explainable AI

## Acknowledgements

The team thanks the supervisor, Prof. Yuanzhu Chen, for guidance and for pushing the team to evaluate what it claimed. The team also thanks Capgemini, the industry partner, for its operational view of how network operations teams troubleshoot faults, and Queen's University for the academic programme and for access to its Frontenac high-performance computing cluster, used during development. Finally, the team thanks the National Telecommunication Institute (NTI) and the Ministry of Communications and Information Technology of Egypt, which run the DiGiLiANS Master Program, and the authors of the TelecomTS dataset [@feng2025telecomts] and of the GSMA telecom knowledge graphs [@yang2026kgrel19; @gsma2026orankg] for releasing them publicly.

<!-- PAGEBREAK -->

<!-- TOC -->

<!-- PAGEBREAK -->
