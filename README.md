# FinSight — AI-assisted Machine Learning Analytics

**BSAN 750 final group project | Streamlit | Python | scikit-learn | optional OpenAI API**

FinSight is an **educational analysis toolbox for banking and financial analysts who do not know ML**. It guides users from an uploaded **CSV, XLS or XLSX** through data inspection, business question definition, user-confirmed target and predictors, preprocessing, model choice, training/tuning, evaluation/comparison, and a persistent AI advisor. The same model engine works with other tabular datasets.

> **Important:** This is a learning and decision-support prototype, **not** a validated credit-risk platform. Do **not** use it to approve/deny loans or upload unauthorized personal financial records.

## Quickest launch — local computer

You need Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate       # on Windows use: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints. **Start on Step 1 with your CSV/Excel file.** The app requires **no synthetic-data button** and does not automatically upload personal files anywhere. A clearly credited *real-world* macroeconomic CSV is included under `data/` for practicing CSV upload and charts; because it is a time series, it is **not** an appropriate predictive benchmark using the app's random train/test split.

### Turn on the live AI assistant

Until an OpenAI API key is configured, the AI panel displays **transparent, built-in rule-based guidance**, which can explain common model choices. It does **not** claim to be a live large-language-model response.

**Streamlit Community Cloud:** app settings > **Secrets**:

```toml
OPENAI_API_KEY = "your-instructor-managed-key"
# Optional, only if the instructor requires a different model:
# OPENAI_MODEL = "gpt-4o-mini"
```

**Local development:** add the same setting to `.streamlit/secrets.toml` (ignored by Git). **Never upload or commit that file**.

**Privacy:** The live AI receives only column-level metadata, selected settings, aggregate target counts, metrics, and model-level summaries. It does **not** receive raw uploaded rows. Questions typed by users are sent to the AI service when a key is enabled. Financial data should be anonymized or public and used only with permission.

## What the project includes (matching the proposal)

| Requested feature | Where it lives |
|---|---|
| CSV + Excel upload; validation & auto inspection | `app_pages/upload.py`, `src/dataio.py` |
| Missingness, distributions, relationships, scatter/box/correlation | `app_pages/explore.py` |
| Plain-English business objective; suggested learning type and targets | `app_pages/problem.py`, `src/dataio.py` |
| Predictor selection + ID/leakage/fairness warnings; train-only imputing/encoding/scaling | `app_pages/prepare.py`, `src/modeling.py` |
| Regression: Linear, Ridge, Lasso, CART, Random Forest | `app_pages/models.py`, `src/modeling.py` |
| Classification: Logistic, CART, Random Forest, Gradient Boosting (+ KNN) | `src/modeling.py` |
| K-Means: K comparison, silhouette, WSS, segment profiles | `src/modeling.py`, `app_pages/train.py`, `app_pages/results.py` |
| Train/test, cross-validation, optional hyperparameter grid | `app_pages/train.py`, `src/modeling.py` |
| Accuracy/precision/recall/F1/ROC-AUC, ROC/PR charts, confusion matrix, RMSE/MAE/R² | `app_pages/results.py` |
| Cross-model comparison on identical held-out splits | `app_pages/results.py`, `src/state.py` |
| Persistent contextual AI chat + simple fallback explanations | `src/chat.py`, `app_pages/advisor.py`, `app.py` |
| Report and predictions/cluster downloads | `src/report.py`, `app_pages/results.py` |

The model training code computes the actual results: **the AI does not produce metrics or train a statistical model by itself**. Important selections remain under the user's control.

## Files and responsibilities

```text
app.py                    Sidebar, navigation, theme, always-visible AI guide
app_pages/                Eight beginner-friendly pages (see docs/ARCHITECTURE.md)
src/dataio.py             CSV/Excel reading, dataset profiling, problem suggestions
src/modeling.py           ML pipelines, model fitting/tuning/evaluation, clustering
src/knowledge.py          Step-by-step explanations and context-specific warnings
src/chat.py               Session-persistent conversation; optional OpenAI API
src/state.py              Shared data, choices, model history, navigation
src/report.py             Downloadable Markdown report
data/                     Credited real-world macroeconomic example CSV (not synthetic)
tests/                    Automated core tests; see docs/TEST_PLAN.md
requirements.txt          Libraries Streamlit installs
.streamlit/config.toml    Accessible theme, readable metric numbers
.env.example              Example key format, no real secrets
```

## Get this onto your existing GitHub + Streamlit website

1. Download the project ZIP and extract it (you should see an `app.py`, `requirements.txt`, `app_pages/`, and `src/` inside).
2. **Recommended: GitHub Desktop.** Clone your existing `evelynbpt/BankSense-AI` repo. Create a new branch, e.g. `fin-sight-rebuild`.
3. **Copy the CONTENTS of `FinSight_AI_Project`** into the cloned repo, not the parent folder. Choose **Replace** for existing files.
4. Remove the old, unrelated `advisor.py`, `business.py`, `inspect_data.py`, `model_choice.py`, `modeling.py`, `sample_data.py`, `train.py`, `upload.py`, etc. from the repository *root*. They are leftovers from earlier accidental folder uploads. Also remove old `app_pages/business.py`, `app_pages/inspect_data.py`, `app_pages/model_choice.py` and old `src/guide.py`, `src/insights.py`, `src/sample_data.py` if present; they aren't used by the rebuilt app.
5. Verify `app.py` and `requirements.txt` are at the **top level**. Commit changes, push the branch, open a Pull Request, and merge into `main` after team review.
6. Your existing Streamlit Cloud deployment should redeploy after a merge into `main`. If deploying afresh, set repository to `evelynbpt/BankSense-AI`, branch `main`, and main file `app.py`.
7. On Streamlit, add the private `OPENAI_API_KEY` secret **only when provided by your professor**. Test the full workflow and chat.

If you're unsure about the GitHub steps, see `docs/ARCHITECTURE.md` and `docs/TEST_PLAN.md` or ask the assistant to walk you through **one click at a time**.

## Example datasets from your proposal (external, not bundled)

- [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) — **classification**; 30,000 credit accounts, predict payment default; has an Excel `.xls` file. When reading the original spreadsheet, choose the sheet with actual data and remove metadata/header rows if needed. Exclude `ID` from predictors.
- [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) — **classification**; term deposit subscription. Its CSV is often semicolon-delimited. Consider whether a feature such as call duration would be known at actual prediction time.
- [UCI Real Estate Valuation](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set) — **regression**; house price. Choose the house-price column as target; exclude the index/number column.
- **Clustering:** use numeric customer financial measurements from the credit card dataset, without labels or identifiers.

Download those datasets from their original host, inspect any data licensing/usage terms, and upload the data into Streamlit. The app does not make automatic third-party downloads, which avoids hidden network dependencies and protects your choice of data.

## Known limitations — be transparent in the demo

- Random train/test splitting: good for many independent rows, **not suitable for time series, grouped customers, or future forecasting without temporal or group-aware validation**.
- No formal fairness, drift, calibration, privacy, causal inference, lending compliance, or production security certification.
- OpenAI-based advice requires a supplied key and a working connection; built-in advice is more limited and explicitly identified as such.
- Suggesting a target or feature by column name is an educational **heuristic**; humans must confirm which outcomes and predictors are appropriate.
- Even with a held-out test set, comparing many models repeatedly can lead to selecting based on test-set noise. Consider external validation for serious use.
- Large files and many categories can be computationally expensive; the prototype caps uploads at 30 MB, 100k rows and 200 columns.

## Automated tests

```bash
pip install pytest
pytest -q
```

See `docs/TEST_PLAN.md` for multi-dataset manual verification. Models' expected scores are **not hard-coded**.

## Who should build what?

- **Owner 1 (app):** `app.py`, upload, UI, deploy.
- **Owner 2 (data):** `src/dataio.py`, inspect and prepare pages, validation.
- **Owner 3 (models):** `src/modeling.py`, models/train/results, tests.
- **Owner 4 (AI):** `src/chat.py`, `app_pages/advisor.py`, prompt design, privacy checks.

Each teammate should use their own branch and a pull request. Ask *why* code changes were made, run tests, and review changes before merging.

## Attribution

The `data/macroeconomic_example_REAL.csv` file is the real `statsmodels.datasets.macrodata` dataset (United States quarterly macroeconomic series based on original Stata Press data; available through the `statsmodels` library). It is included for **file upload and exploratory demonstrations**, not a causal or forecast benchmark. UCI datasets remain credited at the links above.
