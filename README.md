# Agentic RAG & Multi-LLM Intelligence Engine

An end-to-end LLM agent pipeline that ingests real-time tech/AI news from RSS feeds,
classifies sentiment using a fine-tuned DistilBERT model, and generates a structured
daily digest via a locally-hosted LLM (Llama 3.2 via Ollama).

## How it works
1. Pulls headlines from configured RSS feeds
2. Runs each headline through a fine-tuned DistilBERT sentiment classifier (0.54 macro-F1)
3. Passes results to Llama 3.2 (via Ollama) for summarization and categorization
4. Outputs a structured daily digest

## Tech stack
Python, PyTorch, Hugging Face Transformers, Ollama, MLflow

## Setup
[your actual install/run steps here]

## Results
- Sentiment classifier: 0.54 macro-F1 on 800+ labeled tweets, 3-class classification
- All experiments tracked with MLflow
