"""
Полное тестирование двухэтапного LLM-классификатора (Block 1).

Тестирует:
  - Stage 1: Векторный пре-отбор (SBERT + pgvector) — ТОП-K кандидатов
  - Stage 2: LLM-резолвинг через каскад провайдеров (YandexGPT и т.д.)
  - Кэширование в DynamicAlias
  - Ontology expansion (создание новых слагова)

Запуск: python test_enrichment_live.py
"""

import asyncio
import logging
import time

from sentence_transformers import SentenceTransformer
from data.session import global_init, create_session
from config import config
from app.ai_profiler.dynamic_enrichment import get_tag_enricher
from app.ai_profiler.contextual_adapter import get_contextual_adapter
from app.ai_profiler.providers import build_cascade
from data.ai import DynamicAlias

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("test_enrichment")


TEST_TAGS = [
    ("люблю слушать реп", "hip_hop / music_genres", "direct"),
    ("кататься на скейте", "sports_active_life / extreme", "indirect"),
    ("играю в ксочку", "competitive_gaming", "direct"),
    ("делаю 3d модельки", "3d_modeling", "direct"),
    ("читаю фантастику", "fantasy / science_fiction_lit", "direct"),
    ("пишу на пайтоне", "backend_python", "direct"),
    ("гоняю в доту", "competitive_gaming", "direct"),
    ("смотрю аниме", "animation_films", "indirect"),
    ("варю кофе", "home_lifestyle", "indirect"),
    ("играю на гитаре", "music_production", "indirect"),
    ("бегаю по утрам", "fitness_training", "direct"),
    ("собираю лего", None, "create"),
    ("вязание крючком", None, "create"),
    ("каллиграфия", None, "create"),
    ("", None, "empty"),
    ("a", None, "too_short"),
]


def test_stage1_vector_retrieval(enricher, db):
    print("\n" + "=" * 80)
    print("СТЕЙДЖ 1: ВЕКТОРНЫЙ ПРЕ-ОТБОР КАНДИДАТОВ (SBERT + pgvector)")
    print("=" * 80)

    adapter = get_contextual_adapter()
    sbert = SentenceTransformer(
        config.SBERT_MODEL_NAME, model_kwargs={"local_files_only": True}
    )

    for raw_tag, _expected, case_type in TEST_TAGS:
        if not raw_tag or len(raw_tag.strip()) < 2:
            continue

        norm_tag = raw_tag.strip().lower()
        enriched = adapter.enrich_text(norm_tag).enriched
        vec = sbert.encode([enriched], convert_to_numpy=True)[0].tolist()
        candidates = enricher._retrieve_top_k_candidates(db, vec)

        print(f"\n  Tag: '{raw_tag}' [{case_type}]")
        print(f"  Candidates: {len(candidates)}")
        for i, c in enumerate(candidates[:5], 1):
            slug = c['slug']
            name = c['name']
            sim = c['similarity']
            print(f"    {i}. slug={slug:30s} name={name:25s} sim={sim:.4f}")
        if len(candidates) > 5:
            print(f"    ... and {len(candidates) - 5} more")


def test_stage2_llm_resolution():
    print("\n" + "=" * 80)
    print("STAGE 2: LLM RESOLUTION (PROVIDER CASCADE)")
    print("=" * 80)

    cascade = build_cascade()
    provider_names = [p.provider_name for p in cascade.providers]
    print(f"  Providers in cascade: {provider_names}")

    if not cascade.providers:
        print("  No providers configured — LLM stage unavailable")
        return

    sample_candidates = [
        {"slug": "hip_hop", "name": "Hip-hop", "path": "/music_audio/music_genres/hip_hop", "depth": 2, "similarity": 0.72},
        {"slug": "rock_music", "name": "Rock", "path": "/music_audio/music_genres/rock_music", "depth": 2, "similarity": 0.65},
        {"slug": "electronic_music", "name": "Electronic music", "path": "/music_audio/music_genres/electronic_music", "depth": 2, "similarity": 0.60},
        {"slug": "jazz", "name": "Jazz", "path": "/music_audio/music_genres/jazz", "depth": 2, "similarity": 0.55},
        {"slug": "music_production", "name": "Music production", "path": "/creativity_art/music_production", "depth": 1, "similarity": 0.50},
        {"slug": "classical_music", "name": "Classical music", "path": "/music_audio/music_genres/classical_music", "depth": 2, "similarity": 0.48},
        {"slug": "competitive_gaming", "name": "Esports", "path": "/gaming/competitive_gaming", "depth": 1, "similarity": 0.30},
        {"slug": "backend_python", "name": "Python ecosystem", "path": "/it_development/backend_dev/backend_python", "depth": 2, "similarity": 0.25},
        {"slug": "fantasy", "name": "Fantasy", "path": "/literature_reading/reading_genres/fantasy", "depth": 2, "similarity": 0.20},
        {"slug": "drama", "name": "Drama", "path": "/cinema_video/cinema_genres/drama", "depth": 2, "similarity": 0.15},
    ]

    test_prompts = [
        ("люблю слушать реп", "matched", "hip_hop"),
        ("играю в ксочку", "matched", "competitive_gaming"),
        ("катаюсь на сноуборде", "matched", None),
        ("собираю лего", "create", None),
        ("вязание крючком", "create", None),
    ]

    for raw_tag, expected_status, expected_slug in test_prompts:
        print(f"\n  Tag: '{raw_tag}'")
        top3 = [c['slug'] for c in sample_candidates[:3]]
        print(f"  Candidates passed: {len(sample_candidates)}, top-3: {', '.join(top3)}")

        start = time.time()
        result = asyncio.run(cascade.classify(raw_tag, sample_candidates))
        elapsed = time.time() - start

        if result is None:
            print(f"  All providers failed (timeout/error)")
            continue

        status = result.get("status")
        provider = result.get("provider", "?")
        confidence = result.get("confidence", 0.0)
        slug = result.get("slug") or result.get("suggested_slug", "?")
        reason = result.get("reason", "")

        print(f"  Provider: {provider}  ({elapsed:.1f}s)")
        print(f"  Status: {status}")
        print(f"  Slug: {slug}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Reason: {reason}")

        if expected_status == "matched" and expected_slug:
            if slug == expected_slug:
                print(f"  Matches expected: {expected_slug}")
            else:
                print(f"  Expected {expected_slug}, got {slug}")
        if expected_status == "create":
            print(f"  Ontology expansion: new slug '{slug}' proposed")


def test_full_pipeline(enricher, db):
    print("\n" + "=" * 80)
    print("FULL PIPELINE: STAGE 1 -> STAGE 2 -> CACHE")
    print("=" * 80)

    cascade = build_cascade()
    adapter = get_contextual_adapter()
    sbert = SentenceTransformer(
        config.SBERT_MODEL_NAME, model_kwargs={"local_files_only": True}
    )
    provider_line = ", ".join(p.provider_name for p in cascade.providers)
    print(f"  Providers: [{provider_line}]")

    for raw_tag, expected, case_type in TEST_TAGS:
        if not raw_tag or len(raw_tag.strip()) < 2:
            print(f"\n  Tag: '{raw_tag}' [{case_type}] — skip")
            continue

        print(f"\n  Tag: '{raw_tag}' [{case_type}]")
        print(f"  Expected: {expected}")

        norm_tag = raw_tag.strip().lower()
        existing = db.query(DynamicAlias).filter_by(raw_tag=norm_tag).first()
        if existing:
            db.delete(existing)
            db.commit()

        start = time.time()

        enriched = adapter.enrich_text(norm_tag).enriched
        vec = sbert.encode([enriched], convert_to_numpy=True)[0].tolist()
        candidates = enricher._retrieve_top_k_candidates(db, vec)

        print(f"  [Stage 1] Candidates: {len(candidates)}")
        if candidates:
            top3 = []
            for c in candidates[:3]:
                top3.append(f"{c['slug']}({c['similarity']:.3f})")
            print(f"  [Stage 1] Top-3: {', '.join(top3)}")

        if cascade.providers and candidates:
            llm_result = asyncio.run(cascade.classify(norm_tag, candidates))
            if llm_result:
                provider = llm_result.get("provider", "?")
                status = llm_result.get("status")
                slug_res = llm_result.get("slug") or llm_result.get("suggested_slug", "?")
                confidence = llm_result.get("confidence", 0.0)
                reason = llm_result.get("reason", "")
                print(f"  [Stage 2] {provider} -> status={status}, slug={slug_res}, conf={confidence:.3f}, reason='{reason}'")
            else:
                print(f"  [Stage 2] All providers failed")
        else:
            print(f"  [Stage 2] Skipped (no providers or candidates)")

        slug = enricher.resolve_tag_to_slug(db, raw_tag, fallback_to_enrichment=True)
        elapsed = time.time() - start

        if slug:
            print(f"  Result: {slug}  ({elapsed:.2f}s)")
            alias = db.query(DynamicAlias).filter_by(raw_tag=norm_tag).first()
            if alias:
                print(f"  Cache: source={alias.source}, confidence={alias.confidence:.3f}")
        else:
            print(f"  Failed to resolve  ({elapsed:.2f}s)")


def test_cache_behaviour(enricher, db):
    print("\n" + "=" * 80)
    print("CACHE TEST: REPEATED QUERY IN O(1)")
    print("=" * 80)

    test_tag = "люблю слушать реп"
    norm_tag = test_tag.strip().lower()

    cached = db.query(DynamicAlias).filter_by(raw_tag=norm_tag).first()
    if cached:
        db.delete(cached)
        db.commit()
        print(f"  Cleaned existing cache for '{test_tag}'")

    print(f"\n  First query '{test_tag}':")
    start = time.time()
    slug1 = enricher.resolve_tag_to_slug(db, test_tag, fallback_to_enrichment=True)
    t1 = time.time() - start
    print(f"    Result: {slug1}  ({t1:.2f}s)")

    print(f"  Second query '{test_tag}':")
    start = time.time()
    slug2 = enricher.resolve_tag_to_slug(db, test_tag, fallback_to_enrichment=True)
    t2 = time.time() - start
    print(f"    Result: {slug2}  ({t2:.4f}s)")

    if t2 < t1 / 5:
        print(f"  Cache works: repeat {t1/t2:.0f}x faster")
    else:
        print(f"  Cache may not work (t1={t1:.2f}s vs t2={t2:.2f}s)")


def main():
    global_init(config.DATABASE_URL)
    db = create_session()
    enricher = get_tag_enricher()

    print("=" * 80)
    print("TWO-STAGE LLM CLASSIFIER TEST (BLOCK 1)")
    print(f"  Top-K: {config.LLM_CLASSIFIER_TOP_K}")
    print(f"  Timeout: {config.LLM_CLASSIFIER_TIMEOUT}s")
    print(f"  Ollama: {'on' if config.OLLAMA_ENABLED else 'off'}")
    print("=" * 80)

    try:
        test_stage1_vector_retrieval(enricher, db)
    except Exception as e:
        logger.exception("Stage 1 failed: %s", e)

    try:
        test_stage2_llm_resolution()
    except Exception as e:
        logger.exception("Stage 2 failed: %s", e)

    try:
        test_full_pipeline(enricher, db)
    except Exception as e:
        logger.exception("Full pipeline failed: %s", e)

    try:
        test_cache_behaviour(enricher, db)
    except Exception as e:
        logger.exception("Cache test failed: %s", e)

    db.close()
    print("\nTests complete")


if __name__ == "__main__":
    main()
