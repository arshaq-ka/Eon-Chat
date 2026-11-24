# EON Chat

EON Chat is a Streamlit chat app using a local Ollama model and LangGraph. Start
Ollama locally before launching the app.

## Setup

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2:latest
ollama pull nomic-embed-text
python -m streamlit run frontend.py
```

Set `OLLAMA_MODEL` and `OLLAMA_BASE_URL` in `.env` to select the local model and
Ollama server. The URL defaults to `http://localhost:11434`. Tool calling works
with models such as llama3.1, llama3.2, qwen2.5, qwen3, and mistral-nemo. The app
checks the configured model at startup and uses a ReAct prompt fallback when
tool calling is unavailable.

## Tools

The agent can call these tools during a conversation:

| Tool | Purpose | Example prompt |
| --- | --- | --- |
| `get_current_news` | Recent headlines via Google News RSS; optionally uses GNews when `GNEWS_API_KEY` is set. | “Latest AI news” |
| `get_weather` | Current conditions and up to seven forecast days via Open-Meteo. | “What's the weather in Mangalore?” |
| `calculator` | Safe arithmetic and `sqrt`, `log`, `sin`, `cos`, and `tan`. | “What is 18% of 4500?” |
| `get_current_datetime` | Current date and time for an IANA timezone. | “What time is it in Asia/Kolkata?” |
| `web_search` | Top five web results via DuckDuckGo. | “Search the web for recent Python releases.” |
| `purchase_stock` | Submit a whole-share market buy order to an Alpaca paper-trading account. | “Buy 2 shares of AAPL.” |

`GNEWS_API_KEY` is optional; when absent, news uses Google News RSS without a
key. Stock purchases require `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` from an
Alpaca paper-trading account. The tool uses Alpaca's paper endpoint only; it
does not support live trading, selling, fractional shares, or limit orders.
Keep credentials in `.env`, not in source control.

## Document Q&A (RAG)

Attach PDF or DOCX files using the plus button in the chat composer (up to
200 MB per file). When you send files with a prompt, they are indexed before
the agent responds. Files attached without a prompt are indexed in the
background, so you can keep chatting while they are processed. The app extracts
and chunks the text, creates embeddings with the Ollama model named by
`OLLAMA_EMBEDDING_MODEL` (default `nomic-embed-text`), and stores the index in
the local `rag_index/` directory. The index persists across restarts and is
excluded from Git. Ask questions about the uploaded files; the agent searches
the index and cites the source filename (for example, “Summarize the deployment
steps in my uploaded guide”). Re-uploading a changed file with the same name
replaces its indexed content.

## Streaming

Assistant responses stream into the chat as Ollama generates them. Only stock
purchase calls pause for approval. The review shows the ticker and quantity;
approval submits the paper order, while rejection prevents that order from
being submitted. Other tools continue without an approval prompt. This
purchase-only approval step works with native tool calling and the ReAct
fallback.

## Tests

Run the tool tests with:

```powershell
python -m pytest
```
