"""Read-only lexical retrieval over provenance-labelled public training cases."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from langchain_core.documents import Document

from agentic_rag.math_taxonomy import classify_math_text, tokenize_math


def retrieve_public_cases(queries: list[str], path: str, top_k: int) -> list[Document]:
    if not path or top_k < 1:
        return []
    database = Path(path).resolve()
    if not database.is_file():
        return []
    # SQL sees only train rows. Test questions and answer labels are never stored here.
    by_key = {}
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=.2) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        for query in queries[:4]:
            terms = list(dict.fromkeys(tokenize_math(query)))[:48]
            if not terms:
                continue
            match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
            results = connection.execute("""
                SELECT q.*, bm25(question_search) AS lexical_score
                FROM question_search JOIN questions q USING(problem_sha256)
                WHERE question_search MATCH ? AND q.split='train'
                  AND NOT EXISTS(SELECT 1 FROM heldout h WHERE h.problem_sha256=q.problem_sha256)
                ORDER BY lexical_score LIMIT ?
            """, (match, min(top_k, 8)))
            for row in results:
                item = dict(row)
                key = item["problem_sha256"]
                if key not in by_key or item["lexical_score"] < by_key[key]["lexical_score"]:
                    by_key[key] = item
    documents = []
    for item in sorted(by_key.values(), key=lambda row: row["lexical_score"])[:top_k]:
        options = [choice for group in json.loads(item["options_json"]) for choice in group]
        analysis = json.loads(item["analysis_json"])
        content = ("Public training example, not the current problem or a textbook theorem.\n"
            + item["question"] + "\n" + "\n".join(choice["aoVal"]+". "+choice["content"] for choice in options)
            + "\nReference choice: "+item["reference_answer"]+"\nPublisher analysis: "+"\n".join(analysis))
        documents.append(Document(page_content=content[:9000], metadata={
            "chunk_id": "tal-train-"+item["problem_sha256"], "source_type": "public_training_example",
            "source": "https://github.com/math-eval/TAL-SCQ5K/blob/"+item["revision"]
                      +"/ch_single_choice_constructed_5K/ch_single_choice_train_3K.jsonl#"+item["source_id"],
            "source_id": item["source_id"], "split": "train", "revision": item["revision"],
            "chapter": classify_math_text(item["question"]).chapter,
            "retrieval_channel": "public_fts5", "channel_score": -item["lexical_score"],
            "untrusted_external_content": True,
        }))
    return documents
