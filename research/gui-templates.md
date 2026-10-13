# GUI options

Researched 2026-10-13. Based on search-result summaries and the Streamlit docs page for `CheckboxColumn`; no framework was installed or tried.

## What the GUI has to do

1. **Review questions**: a table of generated questions with the evidence beside each, editable text, and an approve/reject tick per row.
2. **Label judge decisions**: show a key point and an answer, record supported / not supported.
3. **Show results**: metric tiles with confidence intervals, breakdown tables, a per-question drill-down.
4. **Compare two runs**: side-by-side metrics with the paired difference.

5. **Set up and start jobs** (added in PRD revision 3): forms for keys and settings, file upload for material, buttons that start generation, runs and scoring with a progress display.

All of these are tables, forms, uploads and a few charts over local files. No login, no multi-user state. Streamlit has built-in widgets for each (`st.form`, `st.file_uploader`, `st.multiselect`, `st.progress`); because it reruns the page script on every interaction, long jobs are started as separate processes (architecture 3.14).

## Recommendation: Streamlit

- `st.data_editor` with `st.column_config.CheckboxColumn` gives the review table directly: editable cells, a tick column, other columns locked with `disabled=`. The edited table comes back as a dataframe to save.
- Multi-page apps, metric tiles (`st.metric`), tables and charts are built in.
- Pure Python, one dependency, runs with `streamlit run`.
- Known limits: the whole script reruns on each interaction, and layout control is coarse. Neither matters at 210 rows on one machine.

Docs: https://docs.streamlit.io/develop/api-reference/data/st.column_config/st.column_config.checkboxcolumn

## Alternatives

| Framework | Fits when | Why not first choice here |
|---|---|---|
| Gradio | Wrapping a single model in a demo | Weak for multi-page tools and editable tables |
| NiceGUI | Admin panels, forms, finer control over events and state (built on FastAPI, Vue, Quasar) | More code for the same tables; worth it only if Streamlit's reruns become a problem |
| Reflex | Long-lived production web app (compiles to React) | Steeper learning curve, far more than this tool needs |
| Dash | Analytics-heavy dashboards | Callback wiring is heavier than Streamlit for forms and review |
| Tkinter / Electron (named in the old architecture) | Desktop app | Tkinter has no good editable table; Electron means a second language and a build chain |

## Existing tools worth looking at before building

| Tool | What it offers | Status |
|---|---|---|
| ragvue (PyPI) | Streamlit UI for RAG eval history, comparison and diagnostics | Seen in search summary only; a possible layout reference |
| open-rag-eval (Vectara, Apache 2.0) | Reference-free RAG eval with a local or web results viewer; metrics built on UMBRELA and AutoNuggetizer | Seen in search summary only |
| RAGLint (MIT) | Metrics plus a dashboard launched from the CLI | README is marketing-heavy; verify before relying on it |
| promptfoo, Arize Phoenix, Argilla | Results viewer, eval/observability UI, annotation tool | Not covered by the search; not checked |

If one of these already shows results well enough, exporting our scores in its format would be less work than building the results pages. This was not investigated.

## Sources

- https://docs.streamlit.io/develop/api-reference/data/st.column_config/st.column_config.checkboxcolumn
- https://discuss.streamlit.io/t/data-editor-checkbox-on-change/84446/2
- https://www.bitdoze.com/md/streamlit-vs-nicegui.md
- https://wikidocs.net/352261
- https://reflex.dev/migration/low-code/ (vendor page)
- https://pypi.org/project/ragvue/
- https://www.sourcepulse.org/projects/2166998 (open-rag-eval listing)
- https://github.com/raglint-dev/raglint
