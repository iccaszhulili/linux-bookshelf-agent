# Linux Bookshelf Agent

A RAG (Retrieval-Augmented Generation) chatbot with conversation memory, served via a Gradio web UI. It answers questions about Linux topics using your personal documentation as the knowledge base.

## Architecture

- **LLM**: DeepSeek (`deepseek-chat`) via LangChain
- **Embeddings**: Ollama (`nomic-embed-text`) running locally
- **Vector Store**: Qdrant (`localhost:6333`, collection `bookshelf_docs`)
- **Conversation Memory**: Mem0 with Qdrant backend (`localhost:6333`, collection `mem0`)
- **Orchestration**: LangGraph (retrieve docs -> retrieve memory -> chatbot)
- **Web UI**: Gradio ChatInterface

## Files

| File | Description |
|------|-------------|
| `chatbot.py` | Gradio web UI that serves the chatbot |
| `mem0_bookshelf.py` | LangGraph pipeline: RAG retrieval + Mem0 memory + LLM |
| `ingest.py` | Incrementally embeds `.md` docs from `DOCS_PATH` into Qdrant |

## Prerequisites

- Python 3.13+
- Ollama installed and running (`ollama serve`)
- Qdrant running on `localhost:6333` (see below)
- DeepSeek API key

## Setup

1. Start Qdrant (Docker):

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

2. Install dependencies:

```bash
pip install gradio mem0ai qdrant-client litellm langchain langgraph langchain-core langchain-qdrant langchain-ollama python-dotenv
```

3. Install spaCy model (required by Mem0):

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

4. Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

5. Create a `.env` file:

```
DOCS_PATH=/path/to/your/markdown/docs
DEEPSEEK_API_KEY=your-deepseek-api-key
```

6. Ingest your documents:

```bash
python ingest.py
```

7. Run the chatbot:

```bash
python chatbot.py
```

Open http://localhost:7860 in your browser.

## Systemd Service

To run as a service, create `/etc/systemd/system/gradio-chatbot.service`:

```ini
[Unit]
Description=Gradio ChatBot Service
After=network.target ollama.service docker.service
Requires=ollama.service

[Service]
Type=simple
User=lizhu
WorkingDirectory=/home/lizhu/linux_bookshelf_agent
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/python3 /home/lizhu/linux_bookshelf_agent/chatbot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gradio-chatbot.service
```

## Usage

- Ask questions about Linux topics in the chat interface
- Type `/memory` to view the conversation memory context from your last query
- Memory persists across conversations in Qdrant (collection `mem0`)
