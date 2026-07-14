#!/usr/bin/env python3
"""Диагностика разрешения тегов — БЕЗ SBERT (только keyword + TF-IDF).

Запуск:
  docker compose exec app python tools/test_resolution.py
"""

import logging
import os
import sys

for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3", "filelock"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import session as db_session
from config import config
from app.ai_profiler.dynamic_enrichment import get_tag_enricher
from data.interest_hierarchy import InterestHierarchyNode

TEST_CASES = [
    ("музыка", "music_audio"),
    ("спорт", "sports_active_life"),
    ("программирование", "it_development"),
    ("футбол", "football"),
    ("python", "backend_python"),
    ("написание кода", "it_development"),
    ("слушаю музыку", "music_audio"),
    ("бэкенд", "backend_dev"),
    ("бег трусцой", "sports_active_life"),
    ("web dev", "frontend_dev"),
    ("нейросети", "machine_learning"),
    ("data science", "data_science"),
]


def main():
    print("=" * 60)
    print("  ДИАГНОСТИКА (KEYWORD + TF-IDF, БЕЗ SBERT)")
    print("=" * 60)

    db_session.global_init(config.SQLALCHEMY_DATABASE_URI)
    sess = db_session.create_session()
    try:
        enricher = get_tag_enricher()
        enricher._cascade = None  # disable LLM

        print("\nTF-IDF index building...")
        enricher._build_tfidf(sess)
        from app.ai_profiler.dynamic_enrichment import _tfidf_slugs
        print(f"TF-IDF: {len(_tfidf_slugs or [])} nodes indexed")

        print(f"\n{'TAG':<25s} {'KEYWORD':<25s} {'TFIDF_TOP1':<25s} {'EXPECTED':<25s} STATUS")
        print("-" * 100)

        ok = wrong = 0
        for tag, expected in TEST_CASES:
            norm = tag.strip().lower()
            kw = enricher._keyword_match(sess, norm)
            tfidf_top = enricher._tfidf_search(norm, top_k=1)
            tfidf_slug = tfidf_top[0]["slug"] if tfidf_top else "-"

            # Result = keyword or tfidf
            result = kw or tfidf_slug
            if result == "-":
                result = None

            status = "OK" if result == expected else "FAIL"
            if result == expected:
                ok += 1
            else:
                wrong += 1

            print(f"{tag:<25s} {str(kw or '-'):<25s} {str(tfidf_slug):<25s} {expected:<25s} {status}")

        print("-" * 100)
        print(f"OK: {ok} / {ok + wrong}   FAIL: {wrong}")

        if wrong:
            print("\n=== DEBUG FAILED TESTS ===")
            from app.ai_profiler.semantic_ontology import SEMANTIC_ONTOLOGY
            for tag, expected in TEST_CASES:
                norm = tag.strip().lower()
                kw = enricher._keyword_match(sess, norm)
                tfidf_top = enricher._tfidf_search(norm, top_k=1)
                tfidf_slug = tfidf_top[0]["slug"] if tfidf_top else None
                result = kw or tfidf_slug
                if result != expected:
                    entry = SEMANTIC_ONTOLOGY.get(expected, {})
                    node = sess.query(InterestHierarchyNode).filter_by(slug=expected).first()
                    print(f"\n  '{tag}' -> kw={kw}, tfidf={tfidf_slug}")
                    print(f"  expected aliases: {entry.get('aliases', [])}")
                    if node:
                        print(f"  DB node: id={node.id} name={node.name}")
                    # Full TF-IDF top 5
                    tfidf5 = enricher._tfidf_search(norm, top_k=5)
                    print(f"  TF-IDF top-5: {[(c['slug'], c['similarity']) for c in tfidf5]}")

    finally:
        sess.close()


if __name__ == "__main__":
    main()
