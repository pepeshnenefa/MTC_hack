"""CSV and batch contract. No network access or model dependencies."""
import csv
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


MAX_ANSWER_CHARS = 10_000
FALLBACK = "Сейчас не удалось подготовить ответ из-за технической ошибки. Пожалуйста, повторите вопрос позже."

class InputError(ValueError):
    pass


def read_questions(path):
    """Preserve input order and IDs as text, including leading zeros."""
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if not header or len(set(header)) != len(header):
                raise InputError("CSV needs a header with unique column names")
            if not {"id", "question"}.issubset(header):
                raise InputError("CSV requires columns id,question (comma separator)")
            id_index, question_index = header.index("id"), header.index("question")
            rows = []
            seen = set()
            for cells in reader:
                if not cells:  # Empty physical lines are not test cases.
                    continue
                if len(cells) != len(header):
                    raise InputError(f"CSV line {reader.line_num}: field count differs from header")
                qid, question = cells[id_index], cells[question_index]
                if not qid.strip() or not question.strip():
                    raise InputError(f"CSV line {reader.line_num}: empty id or question")
                if qid in seen:
                    raise InputError(f"CSV line {reader.line_num}: duplicate ID")
                seen.add(qid)
                rows.append({"id": qid, "question": question})
            return rows
    except (UnicodeError, csv.Error) as exc:
        raise InputError("Invalid UTF-8 or malformed CSV") from exc


def normalize_answer(answer):
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer must be a non-empty string")
    if "\x00" in answer:
        raise ValueError("Answer contains a NUL character")
    normalized = " ".join(answer.split())
    if len(normalized) > MAX_ANSWER_CHARS:
        raise ValueError("Answer exceeds 10000 characters")
    return normalized


def write_answers(path, rows, answers):
    if len(rows) != len(answers):
        raise ValueError("One answer is required for each input row")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate IDs")
    normalized = [normalize_answer(answer) for answer in answers]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # A failed write never leaves a partially written submission.csv.
    fd, temporary = tempfile.mkstemp(prefix=".submission-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(["id", "answer"])
            writer.writerows((row["id"], answer) for row, answer in zip(rows, normalized))
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_answers(path, rows):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != ["id", "answer"]:
            raise ValueError("Output columns must be id,answer")
        output = list(reader)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate input IDs")
    if len(output) != len(rows):
        raise ValueError("Output row count differs from input")
    for index, (cells, row) in enumerate(zip(output, rows), start=1):
        if len(cells) != 2 or cells[0] != row["id"]:
            raise ValueError(f"Output row {index}: ID/order mismatch")
        if normalize_answer(cells[1]) != cells[1]:
            raise ValueError(f"Output row {index}: answer is not normalized")
    return len(output)


def run_batch(rows, answer_one, workers=3, deadline=None, checkpoint=None):
    """Each call receives its own question; checkpoint after every completion.

    The caller's process supervisor enforces the hard wall-clock budget; cancelling
    a future alone cannot stop an in-flight HTTP operation.
    """
    answers = [FALLBACK] * len(rows)
    errors = ['Pending'] * len(rows)
    def work(row):
        try:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Batch deadline')
            return normalize_answer(answer_one(row['question'])), None
        except Exception as exc:
            return FALLBACK, type(exc).__name__
    with ThreadPoolExecutor(max_workers=max(1, min(8, workers))) as pool:
        futures = {pool.submit(work,row):i for i,row in enumerate(rows)}
        for future in as_completed(futures):
            i = futures[future]
            answers[i],errors[i] = future.result()
            if checkpoint is not None:
                checkpoint(answers)
    return answers, errors
