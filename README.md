# LangGraph Workflows

A small, hands-on collection of **agentic workflows built with [LangGraph](https://langchain-ai.github.io/langgraph/)** and [Groq](https://groq.com/)-hosted LLaMA models. Each script builds a small state graph — nodes that read and update a shared state — to do something more structured than a single LLM call. Two of them also ship a [Gradio](https://www.gradio.app/) web UI.

## What's inside

| Script | Pattern | What it does |
| --- | --- | --- |
| [`langgraph_intro.py`](langgraph_intro.py) | **Sequential pipeline** | A three-node graph that runs in order: classify text (News / Blog / Research / Other) → extract entities → summarize. Includes a Gradio UI. |
| [`query_router.py`](query_router.py) | **Conditional routing** | Classifies a question (physics / math / nature / general), then a conditional edge routes it to the matching "expert" prompt so a specialist persona answers. Includes a Gradio UI. |
| [`self_discover.py`](self_discover.py) | **Self-Discover agent** | A four-node pipeline that SELECTs reasoning modules → ADAPTs them to the task → STRUCTUREs them into a JSON plan → REASONs over the plan to answer. Prints each step's state. |

## Tech stack

- Python 3.8+
- [`langgraph`](https://langchain-ai.github.io/langgraph/) (stateful graphs / agent orchestration)
- `langchain` + `langchain-core` + `langchain-groq` (LLM + prompt plumbing)
- `gradio` (web UIs for `langgraph_intro` / `query_router`)
- `python-dotenv` (loads your API key from a `.env` file)
- Groq-hosted models: `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`

> **Version note:** these scripts target the **langchain 0.3 / langgraph 0.2** line. langchain 1.x and langgraph 1.x relocated several import paths used here, so `requirements.txt` pins those ranges.

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/VinayakMokashi/langgraph-workflows.git
   cd langgraph-workflows
   ```

2. **Create and activate a virtual environment** (recommended — keeps the pinned versions isolated)

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Groq API key.** Copy `.env.example` to `.env` and paste in your key
   (get a free one at https://console.groq.com/keys):

   ```bash
   cp .env.example .env
   ```

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

   Each script loads `.env` from its own folder, so the key is found no matter which
   directory you launch from. `.env` is git-ignored and never committed.

## How to run

```bash
python langgraph_intro.py
python query_router.py
python self_discover.py
```

- **`langgraph_intro`** and **`query_router`** print some output and then launch a **Gradio app** at
  **http://127.0.0.1:7860** (press **Ctrl+C** to stop).
- **`self_discover`** prints to the terminal: a direct model answer, then the streamed
  state of each step in the Self-Discover pipeline.

> **Note:** `self_discover` (and each query in `query_router`) makes several model calls.
> On Groq's free tier (6,000 tokens/minute) they may pause to respect rate limits;
> `max_retries` on the model lets them wait and resume rather than crash.

## License

Released under the [MIT License](LICENSE).
