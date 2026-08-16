"""
News Summarizer Agent (with ML Sentiment Analysis)
====================================================
Fetches today's tech/AI news, runs each headline through a fine-tuned
DistilBERT sentiment classifier, then uses a local LLM to summarize
and categorize — producing a structured daily digest.

Pipeline:
  1. Fetch      — pull articles from tech/AI RSS feeds (no API key)
  2. Classify   — run each headline through fine-tuned DistilBERT model
  3. Summarize  — send to local LLM (Ollama/Llama 3.2) for summary + category
  4. Save       — write grouped markdown digest

Requirements:
  pip install -r requirements.txt

  Install Ollama: https://ollama.com → then: ollama pull llama3.2

  Fine-tuned model: train the sentiment classifier first by running:
    cd ../climate_sentiment && python train.py
  This saves the model to ../climate_sentiment/best_model/

Usage:
  python agent.py
"""

import torch
import feedparser
import ollama
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ─── Config ───────────────────────────────────────────────────────────────────

FEEDS = {
    "TechCrunch AI":   "https://techcrunch.com/category/artificial-intelligence/feed/",
    "MIT Tech Review": "https://www.technologyreview.com/feed/",
    "Ars Technica":    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "The Verge":       "https://www.theverge.com/rss/index.xml",
}

CATEGORIES   = ["AI & ML", "Hardware", "Software", "Research", "Industry News", "Other"]
SENTIMENTS   = ["Negative", "Neutral", "Positive"]
MAX_ARTICLES = 8
LLM_MODEL    = "llama3.2"
MODEL_DIR    = "../climate_sentiment/best_model"
MAX_LENGTH   = 128


# ─── Step 1: Fetch ────────────────────────────────────────────────────────────

def fetch_articles(feeds: dict, max_total: int) -> list[dict]:
    articles = []
    for source, url in feeds.items():
        print(f"  Fetching {source}...")
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                articles.append({
                    "source":  source,
                    "title":   entry.get("title", "No title"),
                    "summary": entry.get("summary", entry.get("description", ""))[:500],
                    "link":    entry.get("link", ""),
                })
        except Exception as e:
            print(f"    Could not fetch {source}: {e}")
    return articles[:max_total]


# ─── Step 2: Sentiment Classification (your fine-tuned ML model) ─────────────

def load_sentiment_model(model_dir: str):
    """Load the fine-tuned DistilBERT model trained in climate_sentiment/train.py"""
    print(f"  Loading fine-tuned sentiment model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def classify_sentiment(text: str, tokenizer, model) -> dict:
    """Run a headline through the fine-tuned model and return label + confidence."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs   = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_id = int(torch.argmax(logits))
    label   = SENTIMENTS[pred_id] if pred_id < len(SENTIMENTS) else "Neutral"

    return {
        "sentiment":   label,
        "confidence":  round(probs[pred_id], 3),
    }


# ─── Step 3: Summarize + Categorize (LLM) ────────────────────────────────────

SYSTEM_PROMPT = """You are a tech news assistant.
Given a news article title, snippet, and its sentiment label, return ONLY this exact format:

CATEGORY: <one of: AI & ML, Hardware, Software, Research, Industry News, Other>
SUMMARY: <1-2 sentence plain English summary of what happened and why it matters>"""


def process_article(article: dict) -> dict:
    prompt = (
        f"Title: {article['title']}\n"
        f"Sentiment: {article['sentiment']} (confidence: {article['confidence']})\n"
        f"Snippet: {article['summary']}"
    )
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        raw      = response["message"]["content"].strip()
        category = "Other"
        summary  = article["title"]
        for line in raw.splitlines():
            if line.startswith("CATEGORY:"):
                category = line.replace("CATEGORY:", "").strip()
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
    except Exception as e:
        category = "Other"
        summary  = f"(LLM unavailable: {e})"

    return {**article, "category": category, "llm_summary": summary}


# ─── Step 4: Save Digest ─────────────────────────────────────────────────────

SENTIMENT_EMOJI = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}

def save_digest(articles: list[dict]) -> str:
    date_str    = datetime.now().strftime("%B %d, %Y")
    output_path = f"digest_{datetime.now().strftime('%Y-%m-%d')}.md"

    # Sentiment summary counts
    counts = {s: sum(1 for a in articles if a["sentiment"] == s) for s in SENTIMENTS}

    grouped: dict[str, list] = {cat: [] for cat in CATEGORIES}
    for article in articles:
        cat = article["category"] if article["category"] in CATEGORIES else "Other"
        grouped[cat].append(article)

    lines = [
        f"# 🗞 Tech & AI News Digest",
        f"**{date_str}**\n",
        f"## Sentiment Overview",
        f"🟢 Positive: {counts['Positive']}  |  "
        f"🟡 Neutral: {counts['Neutral']}  |  "
        f"🔴 Negative: {counts['Negative']}\n",
    ]

    for category in CATEGORIES:
        items = grouped[category]
        if not items:
            continue
        lines.append(f"\n## {category}\n")
        for a in items:
            emoji = SENTIMENT_EMOJI.get(a["sentiment"], "🟡")
            lines.append(f"### [{a['title']}]({a['link']})")
            lines.append(f"*{a['source']}* · {emoji} {a['sentiment']} ({a['confidence']})\n")
            lines.append(f"{a['llm_summary']}\n")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ─── Agent Orchestrator ───────────────────────────────────────────────────────

def run_agent():
    print("\n" + "="*50)
    print("  News Summarizer Agent + ML Sentiment")
    print("="*50 + "\n")

    # Step 1: Fetch
    print("[ Step 1 ] Fetching articles...")
    articles = fetch_articles(FEEDS, MAX_ARTICLES)
    print(f"  {len(articles)} articles fetched.\n")

    # Step 2: Classify sentiment with fine-tuned model
    print("[ Step 2 ] Running sentiment classification...")
    try:
        tokenizer, model = load_sentiment_model(MODEL_DIR)
        for article in articles:
            result = classify_sentiment(article["title"], tokenizer, model)
            article.update(result)
            print(f"  {result['sentiment']:8s} ({result['confidence']}) — {article['title'][:55]}...")
    except Exception as e:
        print(f"  Model not found ({e}). Run climate_sentiment/train.py first.")
        print("  Defaulting to Neutral for all articles.\n")
        for article in articles:
            article.update({"sentiment": "Neutral", "confidence": 0.0})
    print()

    # Step 3: Summarize + Categorize with LLM
    print("[ Step 3 ] Summarizing and categorizing with LLM...")
    processed = []
    for i, article in enumerate(articles, 1):
        print(f"  [{i}/{len(articles)}] {article['title'][:55]}...")
        processed.append(process_article(article))
    print()

    # Step 4: Save
    print("[ Step 4 ] Saving digest...")
    output_path = save_digest(processed)
    print(f"  Saved: {output_path}\n")

    print("="*50)
    for a in processed:
        emoji = SENTIMENT_EMOJI.get(a["sentiment"], "🟡")
        print(f"  {emoji} [{a['category']}] {a['title'][:50]}...")
    print(f"\nDigest saved to: {output_path}\n")


if __name__ == "__main__":
    run_agent()
