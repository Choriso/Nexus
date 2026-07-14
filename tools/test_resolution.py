#!/usr/bin/env python3
"""Диагностика разрешения тегов: тест локального пайплайна (без LLM каскада).

Запуск:
  docker compose exec app python tools/test_resolution.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import session as db_session
from config import config
from app.ai_profiler.dynamic_enrichment import (
    DynamicTagEnricher, get_tag_enricher,
    _get_sbert_model, _tfidf_slugs,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)

TEST_CASES = [
    ("музыка", "music_audio"),
    ("спорт", "sports_active_life"),
    ("программирование", "it_development"),
    ("футбол", "football"),
    ("python", "backend_python"),
    ("написание кода", "it_development"),
    ("слушаю музыку", "music_audio"),
    ("играю в футбол", "football"),
    ("изучаю python", "backend_python"),
    ("python разработка", "it_development"),
    ("активный отдых", "sports_active_life"),
    ("чтение книг", "literature_reading"),
    ("бэкенд", "backend_dev"),
    ("танцы", "creativity_art"),
    ("бег трусцой", "sports_active_life"),
    ("готовка еды", "cooking"),
    ("смотрю сериалы", "cinema_video"),
    ("web dev", "frontend_dev"),
    ("data science", "data_science"),
    ("искусственный интеллект", "machine_learning"),
    ("нейросети", "machine_learning"),
    ("занимаюсь спортом каждый день", "sports_active_life"),
    ("пишу код на питоне", "backend_python"),
]


def main():
    print("=" * 60)
    print("  ДИАГНОСТИКА РАЗРЕШЕНИЯ ТЕГОВ (локальный пайплайн)")
    print("=" * 60)

    db_session.global_init(config.SQLALCHEMY_DATABASE_URI)
    sess = db_session.create_session()
    try:
        enricher = get_tag_enricher()

        # Disable LLM cascade — test local pipeline only
        saved_cascade = enricher._cascade
        enricher._cascade = None

        print(f"\nSBERT: {config.SBERT_MODEL_NAME}")
        _get_sbert_model()

        enricher._build_tfidf(sess)
        from app.ai_profiler.dynamic_enrichment import _tfidf_slugs
        print(f"TF-IDF: {len(_tfidf_slugs or [])} nodes indexed")

        print(f"\n{'TAG':<35s} {'HYBRID':<25s} {'KEYWORD':<25s} {'EXPECTED':<25s} STATUS")
        print("-" * 135)

        ok = wrong = 0
        for tag, expected in TEST_CASES:
            norm = tag.strip().lower()
            hybrid = enricher._hybrid_search(sess, norm)
            kw = enricher._keyword_match(sess, norm)
            if hybrid:
                result = hybrid
            elif kw:
                result = kw
            else:
                result = None

            status = "OK" if result == expected else "FAIL"
            if result == expected:
                ok += 1
            else:
                wrong += 1

            print(f"{tag:<35s} {str(hybrid or '-'):<25s} {str(kw or '-'):<25s} {expected:<25s} {status}")

        print("-" * 135)
        print(f"OK: {ok} / {ok + wrong}   FAIL: {wrong}")

        if wrong:
            print("\n=== Иерархия для неудачных тегов ===")
            from data.interest_hierarchy import InterestHierarchyNode
            for tag, expected in TEST_CASES:
                norm = tag.strip().lower()
                hybrid = enricher._hybrid_search(sess, norm)
                kw = enricher._keyword_match(sess, norm)
                result = hybrid or kw
                if result != expected:
                    # Show aliases
                    from app.ai_profiler.semantic_ontology import SEMANTIC_ONTOLOGY
                    entry = SEMANTIC_ONTOLOGY.get(expected, {})
                    print(f"\n  '{tag}' -> resolved={result}, expected={expected}")
                    print(f"    expected aliases: {entry.get('aliases', [])}")
                    # Show nodes matching the expected slug
                    node = sess.query(InterestHierarchyNode).filter_by(slug=expected).first()
                    if node:
                        print(f"    DB node: id={node.id} name={node.name} slug={node.slug} path={node.path}")
                    # Show top-3 SBERT candidates
                    sbert = _get_sbert_model()
                    adapter = enricher._get_adapter()
                    enriched = adapter.enrich_text(tag).enriched
                    vec = sbert.encode([enriched], convert_to_numpy=True)[0].tolist()
                    candidates = enricher._retrieve_top_k_candidates(sess, vec)[:3]
                    print(f"    SBERT top-3: {[(c['slug'], c['similarity']) for c in candidates]}")
                    # Show TF-IDF
                    tfidf = enricher._tfidf_search(norm, top_k=3)
                    print(f"    TF-IDF top-3: {[(c['slug'], c['similarity']) for c in tfidf]}")

        # Restore
        enricher._cascade = saved_cascade

    finally:
        sess.close()


if __name__ == "__main__":
    main()
