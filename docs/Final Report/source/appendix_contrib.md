### Appendix K: Team contributions

The proposal assigned four roles: data preprocessing, feature engineering and the scale ablation; the two classifiers and the SHAP layer; the alignment table and retrieval; and the LLM explanation layer, evaluation and dashboard. {tab:contrib} records the contributions as the members' own reports describe them. The report on the first prototype (12 September 2026) does not attribute individual work, so its components are listed as team work.

Table: Each member's contributions are listed as their reports state them; work that the reports attribute to the ML team as a whole has its own row. {#tab:contrib}

| Member | Contributions |
|---|---|
| Ahmed Mohamed Al Shobaki (team leader) | Retrieval pipeline and its redesign (corpus analysis, per-window narrative index, label-free query with class filter); prompt design and the per-window evidence block; LLM explanation layer with citation normalisation, validation and error handling; Track C ablation, the retrieval design experiment and the evaluation scripts; Streamlit dashboard and deployment; test suite and packaging |
| Farida El Ghabary | Data and models report (14 October 2026); the v3 synthetic training data and the Model 2 recipe (CatBoost with v3 synthetic data, Config B and FFT features) |
| Raneem Alnaghy | Reconstruction of the 33 recording sessions (300 s gap) and the session-level train/test split (credited to Raneem Alnaghy in the ML team's report; first used in the first prototype); the reject option (hard-negative Normal class) and its promotion rule |
| ML team (Farida El Ghabary, Raneem Alnaghy) | Frozen experimental protocol and development steps 01 to 12c on both models (Appendix B); final Layer 2 pipeline and output |
| Rodina Khallaf | 3GPP Release 19 and O-RAN alignment table (matrix v1.2) and its causal-mechanism analysis; Track B explanation-quality evaluation (G-Eval); Track D TeleQnA benchmark evaluation |
| Whole team | First prototype (September 2026): dataset analysis, feature engineering and ablation, first cascade, first alignment table, first retrieval and explanation layer, first Track C ablation, first dashboard |
