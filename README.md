# 01 - RAG with LangChain (Basics)

A minimal, clean Retrieval-Augmented Generation pipeline:
**load -> chunk -> embed -> FAISS index -> retrieve -> grounded answer with citations.**

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python -m src.ingest   # builds ./index
python -m src.rag "How many vacation days can I carry over?"
```

## Key ideas
- `RecursiveCharacterTextSplitter` with overlap keeps context across chunks.
- The prompt forces the model to answer ONLY from context and say "I don't know" otherwise (first line of defence against hallucination).
- Each answer returns the source chunks so you can audit it.
