# Hidden Variables — Member 4: Explainability & Evaluation

**Owner:** Sunshine (Hetvi)  
**Stack:** Python · LangChain · Chroma · Groq (Llama 3.1) · Streamlit

---

## What this module does

| Feature | File | Description |
|---|---|---|
| **Source Highlighting** | `explainability.py` | Every answer is tagged with the exact PDF filename, page number, and paragraph (chunk index within the page), plus a 120-char snippet. |
| **Confidence Score** | `explainability.py` | Cosine distance from Chroma is converted to a human-readable `0–100 %` confidence. `< 55 %` distance → answer shown; `≥ 55 %` → "not found". |
| **Answer Validation** | `explainability.py` | If no chunk is close enough to the query, the LLM is never called and a clear "Answer not found in document" message is returned instead of hallucinating. |
| **PDF Summarisation** | `explainability.py` | `summarize_pdf(path)` sends the first 8 pages to the LLM and returns an Executive Summary, Key Topics list, and Main Conclusions list. |
| **Evaluation Metrics** | `explainability.py` | Tracks Total Queries, Retrieval Accuracy %, Avg Latency (ms), Avg Confidence %, and Avg Relevance Score across sessions in `metrics_log.json`. |
| **Streamlit UI** | `streamlit_ui.py` | Three-tab dashboard: Ask & Explain, PDF Summariser, Eval Metrics. |

---

## How to run

### Prerequisites
Make sure **Sneha's ingestion pipeline** has already been run to populate the vector store:
```bash
cd ../Hidden-Variables-sneha
python main.py
```

### Install
```bash
pip install -r requirement.txt
```

### CLI (quick sanity check)
```bash
python main.py
```

### Streamlit dashboard
```bash
streamlit run streamlit_ui.py
```

---

## Integration with other members

| Member | Their file | How this module connects |
|---|---|---|
| Sneha | `main.py` | Reads the same `./chroma_langchain_db` Chroma directory Sneha's ingestion creates. |
| Vaishnavi | `main.py` (Chroma query) | `ExplainabilityEngine` replaces `similarity_search` with `similarity_search_with_score` to get distances for confidence calculation. |
| Soha | `vectorization.py`, `streamlit_ui.py` | `streamlit_ui.py` here extends Soha's Echo Bot UI pattern into a full three-tab dashboard. The same `filtered_results` logic from `vectorization.py` is formalised into the `_CONF_NOT_FOUND_THRESHOLD` constant. |

---

## Output examples

### Source Highlighting
```
Answer: CNN is used for image processing.
Source: lecture_notes.pdf — Page 17, Paragraph 3
        "…Convolutional Neural Networks (CNN) extract spatial features from images…"
Confidence: 89%
```

### Answer Not Found
```
❌ Answer not found in document.
   (No passage matched with sufficient similarity — retrieval distance ≥ 0.55)
```

### Metrics
```
Total Queries              42
Retrieval Accuracy         88.1%
Avg Confidence             76.4%
Avg Latency                1240 ms
Avg Relevance Score        0.3812
```

---

## File structure
```
Hidden-Variables-sunshine/
├── explainability.py   # Core engine (source highlighting, confidence, validation, summarisation, metrics)
├── streamlit_ui.py     # Three-tab Streamlit dashboard
├── main.py             # CLI entry point
├── requirement.txt
├── README.md
└── metrics_log.json    # Auto-created on first query; persists across sessions
```
