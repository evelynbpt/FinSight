# FinSight manual verification before your BSAN 750 demo

## Core upload and inspection

- [ ] Upload `.csv` with comma delimiter.
- [ ] Upload Bank Marketing CSV using semicolon delimiter and check that columns are correctly separated.
- [ ] Upload `.xlsx` with multiple sheets; choose the intended one.
- [ ] Upload `.xls` (original credit default file); choose the table that contains actual observations.
- [ ] Verify row/column/missing counts, color contrast in Streamlit light and dark browser themes.
- [ ] Check distribution, scatter, boxplot, correlation heatmap, and data-quality warnings.

## Classification demonstration — UCI Default of Credit Card Clients or Bank Marketing

- [ ] Write the business question in everyday English.
- [ ] Verify recommendation is **classification**.
- [ ] Choose the historical default/subscription target and exclude ID, leaked and sensitive predictors when appropriate.
- [ ] Train Logistic Regression, CART, Random Forest, and Gradient Boosting with same predictors and test fraction.
- [ ] Read Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curve, PR curve; confirm results are not copied or simulated.
- [ ] Change probability cutoff; verify FP and FN counts change appropriately.
- [ ] Compare compatible models. Ask AI what false negatives mean; verify it references current computed metrics.

## Regression demonstration — UCI Real Estate Valuation

- [ ] Select house-price column as numeric target and exclude the index/ID column.
- [ ] Train Linear, Ridge, Lasso, CART Regression Tree, Random Forest Regression.
- [ ] Compare RMSE, MAE, R² and baseline RMSE.
- [ ] Look at actual-vs-predicted plot and export the Markdown report.

## Clustering demonstration — credit default financial numeric measurements (without target)

- [ ] Select K-Means and numerical financial predictors only.
- [ ] Run Compare K = 2 … 8; review silhouette and within-cluster SS.
- [ ] Choose K and fit. Confirm cluster sizes add up to data rows.
- [ ] Review group profiles and 2D PCA visualization.
- [ ] Ask AI why choosing the largest K is not automatically best.

## AI / privacy / security

- [ ] Built-in assistant identifies itself as limited when no API key is present.
- [ ] After adding the private API key in Streamlit Secrets, live AI returns context-specific explanations.
- [ ] Chat history stays visible when switching between pages.
- [ ] A new uploaded dataset resets history, preventing outdated metrics from being quoted.
- [ ] `.streamlit/secrets.toml`, `.env`, raw private records are not on GitHub.
- [ ] All models use training-only preprocessing and training-only CV for tuning.
- [ ] Avoid automatic loan decisions and warn about fairness, privacy, causality, and future drift.

## Final submission

- [ ] App link opens without errors and all pages load.
- [ ] GitHub repository has complete source files, README, tests and public example data.
- [ ] 8–10 minute demo rehearsed with a real, attributed UCI dataset.
- [ ] Present both modeling tradeoffs and limitations, not only screenshots.
