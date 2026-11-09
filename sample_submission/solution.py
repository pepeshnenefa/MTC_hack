"""Минимальный сабмит МТС: пример работы с LLM, эмбеддером и реранкером.

Это учебный пример, а НЕ полноценная RAG-система. В частности, он использует
только короткое начало каждого документа, без поиска по всем фрагментам.
"""

import csv
import math
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx


def env(name, default=""):
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def number(name, default, lower, upper, kind=int):
    try:
        return max(lower, min(upper, kind(env(name))))
    except ValueError:
        return default


# Переменные предоставляет платформа. Секреты в ZIP не записывайте.
DATA_DIR = Path(env("DATA_DIR", "data"))
BASE_URL = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")).rstrip("/")
LLM_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY")
EMB_KEY = env("EMBEDDER_API_KEY") or LLM_KEY
RERANK_KEY = env("RERANKER_API_KEY") or LLM_KEY
LLM_MODEL = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "qwen/qwen3.8-27b")
EMB_MODEL = env("EMB_MODEL", "baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", "qwen/qwen3-reranker-8b")
TOP_K = number("TOP_K", 4, 1, 30)
RERANKER_TOP_N = number("RERANKER_TOP_N", 2, 1, 30)
MAX_TOKENS = number("MAX_TOKENS", 2048, 128, 8192)
MAX_WORKERS = number("MAX_WORKERS", 3, 1, 8)
TEMPERATURE = number("TEMPERATURE", 0.0, 0.0, 2.0, float)
CHUNKER = env("CHUNKER", "common")
# PYTHONPATH применяется самим Python, вручную обрабатывать его не нужно.

FALLBACK = "Не могу ответить по доступным фрагментам базы знаний."
WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{4,}")


def find_file(*names):
    for root in (DATA_DIR, Path(__file__).resolve().parent / DATA_DIR, Path.cwd()):
        for name in names:
            path = root / name
            if path.is_file():
                return path
    raise FileNotFoundError(f"Не найдены файлы: {names}")


def words(text):
    return set(WORD.findall(text.lower().replace("ё", "е")))


def excerpt(text):
    # Намеренно очень простой поиск: только первые 420 символов документа.
    part = text[:420]
    if CHUNKER == "common" and len(text) > 420:
        boundary = part.rfind("\n")
        if boundary >= 210:
            part = part[:boundary]
    return " ".join(part.split())


def load_documents(path):
    documents = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.startswith("__MACOSX/") or Path(name).suffix.lower() not in (".md", ".txt"):
                continue
            text = archive.read(name).decode("utf-8-sig", errors="replace")
            if text.strip():
                piece = excerpt(text)
                documents.append((name, piece, words(name + " " + piece)))
    if not documents:
        raise ValueError("База знаний не содержит файлов .md или .txt")
    return documents


def post(client, endpoint, key, body):
    if not BASE_URL or not key:
        raise ValueError("Проверьте переменные адреса API и ключа")
    for attempt in range(2):
        try:
            response = client.post(
                BASE_URL + endpoint,
                headers={"Authorization": "Bearer " + key},
                json=body,
            )
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
            if attempt == 1:
                raise
            time.sleep(0.5)


def embeddings(client, texts):
    """Пример POST /embeddings с моделью из EMB_MODEL."""
    result = post(client, "/embeddings", EMB_KEY, {"model": EMB_MODEL, "input": texts})
    vectors = [None] * len(texts)
    for i, item in enumerate(result["data"]):
        pos = int(item.get("index", i))
        v = [float(x) for x in item["embedding"]]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vectors[pos] = [x / norm for x in v]
    if any(v is None for v in vectors):
        raise ValueError("Не все эмбеддинги получены")
    return vectors


def select_sources(client, question, documents):
    # Этап 1: простое совпадение слов, без полноценного векторного индекса.
    query_words = words(question)
    candidates = sorted(documents, key=lambda d: len(query_words & d[2]), reverse=True)
    candidates = candidates[:min(TOP_K, 4)]  # учебное ограничение: максимум 4 документа

    # Этап 2: демонстрация ранжирования по эмбеддингам.
    try:
        vectors = embeddings(client, [question] + [x[1] for x in candidates])
        scores = [sum(a * b for a, b in zip(vectors[0], v)) for v in vectors[1:]]
        candidates = [candidates[i] for i in sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)]
    except (httpx.HTTPError, KeyError, TypeError, ValueError, IndexError) as exc:
        print(f"Embeddings: {type(exc).__name__}; продолжаем без них", file=sys.stderr)

    # Этап 3: пример POST /rerank.
    try:
        result = post(client, "/rerank", RERANK_KEY, {
            "model": RERANKER_MODEL,
            "query": question,
            "documents": [x[0] + "\n" + x[1] for x in candidates],
            "top_n": min(RERANKER_TOP_N, len(candidates)),
        })
        entries = result.get("results", result.get("data", []))
        indices = [int(item["index"]) for item in entries]
        selected = [candidates[i] for i in indices if 0 <= i < len(candidates)]
        if selected:
            return selected[:RERANKER_TOP_N]
    except (httpx.HTTPError, KeyError, TypeError, ValueError, IndexError) as exc:
        print(f"Reranker: {type(exc).__name__}; продолжаем без него", file=sys.stderr)
    return candidates[:RERANKER_TOP_N]


def generate(client, question, sources):
    """Пример POST /chat/completions с моделью из LLM_MODEL."""
    context = "\n".join(name + ": " + piece for name, piece, _ in sources)
    payload = {
        "model": LLM_MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": "Ты консультант МТС EnergyTool. Ответь по-русски и кратко. Не выдумывай факты, которых нет во фрагментах."},
            {"role": "user", "content": "Фрагменты базы знаний:\n" + context + "\n\nВопрос:\n" + question},
        ],
    }
    for attempt in range(2):
        try:
            result = post(client, "/chat/completions", LLM_KEY, payload)
            answer = result["choices"][0]["message"].get("content", "")
            if isinstance(answer, str) and answer.strip():
                return " ".join(answer.split())  # одна физическая строка CSV
            raise ValueError("LLM вернула пустой ответ")
        except (httpx.HTTPError, KeyError, TypeError, ValueError, IndexError) as exc:
            print(f"LLM попытка {attempt + 1}: {type(exc).__name__}", file=sys.stderr)
    return FALLBACK


def main():
    questions_path = find_file("questions.csv", "train.csv")
    kb_path = find_file("knowledge-base.zip", "knowledge_base.zip", "kb.zip")
    with questions_path.open(encoding="utf-8-sig", newline="") as f:
        questions = [(str(row["id"]).strip(), str(row["question"])) for row in csv.DictReader(f)]
    documents = load_documents(kb_path)
    print(f"Вопросов: {len(questions)}; документов: {len(documents)}", flush=True)

    def work(i, question):
        try:
            with httpx.Client(timeout=90) as client:
                docs = select_sources(client, question, documents)
                return i, generate(client, question, docs)
        except Exception as exc:
            print(f"Вопрос {i + 1}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return i, FALLBACK

    answers = [FALLBACK] * len(questions)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(work, i, question) for i, (_, question) in enumerate(questions)]
        for future in as_completed(futures):
            i, answer = future.result()
            answers[i] = answer

    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        writer.writerows((qid, answers[i]) for i, (qid, _) in enumerate(questions))
    print(f"Готов submission.csv: {len(questions)} ответов", flush=True)


if __name__ == "__main__":
    main()
