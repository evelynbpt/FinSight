# FinSight — How the code works (beginner agenda)

1. **`app.py` (navigation).** It is the starting point; Streamlit runs this file and calls the current page's `render()` function. The sidebar's AI form uses the same chat history on every page.
2. **`app_pages/upload.py` (input).** This calls `src/dataio.read_dataset()` to convert a CSV/Excel file into a `pandas.DataFrame` and stores it as `st.session_state.df`. New uploads reset models/chat history.
3. **`app_pages/explore.py` (inspection).** Displays data profile and Plotly charts. It does not alter the data.
4. **`app_pages/problem.py` (what we want to learn).** Stores `goal`, `task` and `target` in `st.session_state`. Suggestions are simple keyword/column-name hints — not verified answers.
5. **`app_pages/prepare.py` (what the model may use).** Stores `features`. It warns about IDs, missingness and possible target leakage. The actual numeric imputation, categorical encoding and scaling happen in `src/modeling.py` so CV and test data do not leak into preprocessing.
6. **`app_pages/models.py` (which algorithm).** Stores `model_name` based on compatible models.
7. **`app_pages/train.py` (the calculations).** Calls `src/modeling.py` with test fraction, CV folds and optional tuning. Saves a result in `st.session_state.result` and a bounded model `history`. `src/modeling.py` includes sklearn pipelines and all actual metrics.
8. **`app_pages/results.py` (interpretation).** Reads stored model result, plots test metrics, lets users change the classification threshold without retraining, compares compatible models and exports results.
9. **`src/chat.py` + `app_pages/advisor.py` (learning assistant).** `src/chat.py` builds a privacy-limited context from dataset metadata and aggregated results. With an API key it asks an OpenAI model; without a key it returns labeled, rule-based educational explanations. Chat history is in `st.session_state.chat_history`.
10. **`src/state.py` (shared state).** The eight steps share one dataset, choices, model output and conversation while the user navigates.

### Your first safe code edit

Open `app_pages/upload.py`. Change `st.title('1 · Upload your data')` to `st.title('1 · Import a dataset')`. Save, commit, push, and verify the title changes on Streamlit. This affects only the text displayed on the screen.

### Want to change the model list?

Edit `MODELS` in `src/modeling.py` and make sure `make_estimator()` and associated evaluation logic support the new algorithm. `app_pages/models.py` reads the model list automatically.

### Want to change the advice?

Edit the lesson in `src/knowledge.py`, or the `SYSTEM` instruction and simple fallback explanation functions in `src/chat.py`. **Never put actual personal records or API keys into prompts or code.**

### Session-by-session learning schedule

- Session A: Upload and navigation (`app.py`, `upload.py`, `src/state.py`).
- Session B: Inspect and choose columns (`explore.py`, `problem.py`, `prepare.py`).
- Session C: Models, train/test split and pipelines (`src/modeling.py`).
- Session D: Classification vs regression vs clustering scores (`results.py`).
- Session E: Persistent AI chat, connected vs rule-based advice (`src/chat.py`).
- Session F: Run tests, commit to GitHub, deploy and rehearse a finance dataset demo.
