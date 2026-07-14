#!/usr/bin/env python3
"""Диагностика разрешения тегов: пошаговый тест всех стадий пайплайна.

Запуск на сервере:
  docker compose exec app python tools/test_resolution.py

Или внутри контейнера:
  cd /app && python tools/test_resolution.py
"""

import logging
import sys
from typing import Optional

from data import session as db_session
from config import config
from app.ai_profiler.dynamic_enrichment import (
    DynamicTagEnricher, get_tag_enricher,
    _get_sbert_model, _get_cascade,
)
from app.ai_profiler.providers import FailoverCascade
from app.ai_profiler.semantic_ontology import SEMANTIC_ONTOLOGY

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_resolution")

TEST_CASES = [
    # (входной_текст, ожидаемый_слаг, группа)
    # --- Точные алиасы ---
    ("музыка", "music_audio", "exact_alias"),
    ("спорт", "sports_active_life", "exact_alias"),
    ("программирование", "it_development", "exact_alias"),
    ("футбол", "football", "exact_alias"),
    ("python", "backend_python", "exact_alias"),
    # --- Частичные алиасы (фразы) ---
    ("написание кода", "it_development", "partial_alias"),
    ("слушаю музыку", "music_audio", "partial_alias"),
    ("играю в футбол", "football", "partial_alias"),
    ("изучаю python", "backend_python", "partial_alias"),
    ("python разработка", "it_development", "partial_alias"),
    # --- ILIKE / name match ---
    ("активный отдых", "sports_active_life", "keyword_name"),
    ("чтение книг", "literature_reading", "keyword_name"),
    ("бэкенд", "backend_dev", "keyword_name"),
    ("танцы", "creativity_art", "keyword_partial"),
    # --- SBERT / семантика (нет точных алиасов) ---
    ("бег трусцой", "sports_active_life", "semantic"),
    ("готовка еды", "cooking", "semantic"),
    ("смотрю сериалы", "cinema_video", "semantic"),
    ("web dev", "frontend_dev", "semantic"),
    ("data science", "data_science", "semantic"),
    # --- Сложные / граничные ---
    ("искусственный интеллект", "machine_learning", "semantic"),
    ("нейросети", "machine_learning", "semantic"),
    ("занимаюсь спортом каждый день", "sports_active_life", "long_phrase"),
    ("пишу код на питоне", "backend_python", "long_phrase"),
    ("", None, "edge"),
    ("a" * 200, None, "edge"),
]

EXPECTED_MAP = {t[0]: t[1] for t in TEST_CASES}


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def test_cache_layer(enricher: DynamicTagEnricher, sess, tag: str):
    """Проверяет, есть ли тег в кэше DynamicAlias."""
    from data.ai import DynamicAlias
    from app.ai_profiler.dynamic_enrichment import UNRESOLVED_SENTINEL
    norm = tag.strip().lower()
    cached = sess.query(DynamicAlias).filter_by(raw_tag=norm).first()
    if cached:
        status = "UNRESOLVED" if cached.slug == UNRESOLVED_SENTINEL else cached.slug
        return f"CACHED({status})"
    return "MISS"


def test_vector_search(enricher: DynamicTagEnricher, sess, tag: str) -> list[dict]:
    """Тестирует SBERT векторный поиск, возвращает топ-3 кандидата."""
    sbert = _get_sbert_model()
    adapter = enricher._get_adapter()
    enriched = adapter.enrich_text(tag).enriched
    vec = sbert.encode([enriched], convert_to_numpy=True)[0].tolist()
    candidates = enricher._retrieve_top_k_candidates(sess, vec)
    return candidates[:3]


def test_tfidf_search(enricher: DynamicTagEnricher, sess, tag: str) -> list[dict]:
    """Тестирует TF-IDF поиск, возвращает топ-3 кандидата."""
    enricher._build_tfidf(sess)
    return enricher._tfidf_search(tag, top_k=3)


def test_hybrid(enricher: DynamicTagEnricher, sess, tag: str) -> Optional[str]:
    """Тестирует гибридный поиск."""
    return enricher._hybrid_search(sess, tag)


def test_keyword(enricher: DynamicTagEnricher, sess, tag: str) -> Optional[str]:
    """Тестирует keyword fallback через enricher._keyword_match."""
    kw = enricher._keyword_match(sess, tag.strip().lower())
    if kw:
        return kw
    return "NONE"


def test_user_weights(sess, user_id: int):
    """Показывает все веса пользователя."""
    from data.interest_hierarchy import UserInterestGraphWeight, InterestHierarchyNode
    rows = (
        sess.query(UserInterestGraphWeight, InterestHierarchyNode.name, InterestHierarchyNode.slug)
        .join(InterestHierarchyNode, UserInterestGraphWeight.node_id == InterestHierarchyNode.id)
        .filter(UserInterestGraphWeight.user_id == user_id)
        .all()
    )
    if not rows:
        return "  (нет весов)"
    result = []
    for w, name, slug in rows:
        result.append(f"  node_id={w.node_id} slug={slug} name={name} weight={w.weight}")
    return "\n".join(result)


def main():
    print_header("ДИАГНОСТИКА РАЗРЕШЕНИЯ ТЕГОВ")
    print(f"YandexGPT API KEY: {'✓' if config.YANDEX_GPT_API_KEY else '✗'}")

    sess = db_session.create_session()
    try:
        enricher = get_tag_enricher()

        print("\n--- Инициализация SBERT и TF-IDF ---")
        sbert_ok = False
        try:
            _get_sbert_model()
            sbert_ok = True
            print(f"  SBERT: ✓ ({config.SBERT_MODEL_NAME})")
        except Exception as e:
            print(f"  SBERT: ✗ ({e})")

        try:
            enricher._build_tfidf(sess)
            print(f"  TF-IDF: ✓ ({len(enricher._tfidf_slugs or [])} nodes)")
        except Exception as e:
            print(f"  TF-IDF: ✗ ({e})")

        print(f"\n  LLM Cascade: {'✓' if enricher._cascade and enricher._cascade.providers else '✗ (no providers)'}")
        if enricher._cascade:
            for p in enricher._cascade.providers:
                print(f"    - {p.provider_name}")

        print_header("ПОТЕМЕНТНОЕ ТЕСТИРОВАНИЕ")
        stats = {"ok": 0, "wrong": 0, "error": 0}

        for tag, expected, group in TEST_CASES:
            print(f"\n--- [{group:15s}] '{tag}' -> ожидается: {expected or 'None'} ---")
            if not tag or not tag.strip():
                print(f"  SKIP (empty)")
                continue

            try:
                cache = test_cache_layer(enricher, sess, tag)
                print(f"  [1] Кэш: {cache}")

                if sbert_ok:
                    vec_top = test_vector_search(enricher, sess, tag)
                    print(f"  [2] SBERT top-3: {[(c['slug'], c['similarity']) for c in vec_top] or 'нет'}")
                else:
                    print(f"  [2] SBERT: ✗")

                tfidf_top = test_tfidf_search(enricher, sess, tag)
                print(f"  [3] TF-IDF top-3: {[(c['slug'], c['similarity']) for c in tfidf_top] or 'нет'}")

                hybrid_result = test_hybrid(enricher, sess, tag)
                print(f"  [4] Hybrid result: {hybrid_result or 'NONE'}")

                kw = test_keyword(enricher, sess, tag)
                print(f"  [5] Keyword: {kw}")

                result = enricher.resolve_tag_to_slug(sess, tag, fallback_to_enrichment=False, force=True)

                status = "✓" if result == expected else "✗"
                print(f"  >>> ИТОГ: {result or 'None'} {status}")

                if result == expected:
                    stats["ok"] += 1
                elif expected is not None:
                    stats["wrong"] += 1
                else:
                    stats["ok"] += 1

            except Exception as e:
                print(f"  ERROR: {e}")
                stats["error"] += 1
                import traceback
                traceback.print_exc()

        print_header("ВЕСА ПОЛЬЗОВАТЕЛЕЙ")
        from data.user import User as DBUser
        users = sess.query(DBUser).all()
        for u in users:
            print(f"\n  User: {u.id} ({u.name or 'no name'})")
            print(test_user_weights(sess, u.id))

        print_header("СТАТИСТИКА")
        print(f"  Всего тестов: {len(TEST_CASES)}")
        print(f"  Успешно:    {stats['ok']}")
        print(f"  Неверно:    {stats['wrong']}")
        print(f"  Ошибок:     {stats['error']}")

        if stats["wrong"] > 0:
            print_header("НЕВЕРНЫЕ РЕЗУЛЬТАТЫ")
            print("  Проверьте логи выше для каждого случая с ✗")

    finally:
        sess.close()


if __name__ == "__main__":
    main()
