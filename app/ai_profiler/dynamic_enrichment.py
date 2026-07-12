from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Optional

from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.ai_profiler.contextual_adapter import get_contextual_adapter
from app.ai_profiler.providers import FailoverCascade, build_cascade
from config import config
from data.ai import DynamicAlias
from data.interest_hierarchy import InterestHierarchyNode

logger = logging.getLogger(__name__)

UNRESOLVED_SENTINEL = "__UNRESOLVED__"

_sbert_model: Optional[SentenceTransformer] = None
_sbert_model_name: Optional[str] = None
_sbert_model_downloaded = False
_cascade: Optional[FailoverCascade] = None


def _get_sbert_model() -> SentenceTransformer:
    """Возвращает единственный экземпляр SBERT-модели (lazy singleton).

    Пытается загрузить модель локально, при неудаче — скачивает,
    если разрешено конфигом (SBERT_ALLOW_DOWNLOAD).

    Returns:
        Загруженная SBERT-модель.

    Raises:
        RuntimeError: Если модель недоступна ни локально, ни для скачивания.
    """
    global _sbert_model, _sbert_model_name, _sbert_model_downloaded

    if _sbert_model is not None:
        return _sbert_model

    model_name = config.SBERT_MODEL_NAME
    _sbert_model_name = model_name

    try:
        _sbert_model = SentenceTransformer(model_name)
        return _sbert_model
    except Exception as e_local:
        logger.warning(f"[sbert] Local load failed: {e_local}")

    allow_download = config.SBERT_ALLOW_DOWNLOAD
    if not allow_download:
        raise RuntimeError(f"SBERT model '{model_name}' not found locally (downloads disabled)")

    if not _sbert_model_downloaded:
        try:
            _sbert_model = SentenceTransformer(model_name)
        except TypeError:
            _sbert_model = SentenceTransformer(model_name, model_kwargs={"local_files_only": False})
        _sbert_model_downloaded = True
        return _sbert_model

    raise RuntimeError(f"SBERT model '{model_name}' unavailable")


def _get_cascade() -> FailoverCascade:
    """Возвращает единственный экземпляр каскада LLM-провайдеров (lazy singleton).

    Returns:
        Экземпляр FailoverCascade.
    """
    global _cascade
    if _cascade is None:
        _cascade = build_cascade()
    return _cascade


class DynamicTagEnricher:
    """Обогащает и разрешает пользовательские теги в слаг иерархии интересов.

    Двухстадийный процесс:
    Stage 1 — SBERT-векторный поиск top-K кандидатов.
    Stage 2 — LLM-классификация для точного разрешения.
    """

    def __init__(self):
        self.top_k = config.LLM_CLASSIFIER_TOP_K
        self.confidence_threshold = config.LLM_CLASSIFIER_CONFIDENCE_THRESHOLD
        self._cascade = _get_cascade()
        self._local_adapter = None

    def _get_adapter(self):
        """Возвращает контекстуальный адаптер (lazy singleton)."""
        if self._local_adapter is None:
            self._local_adapter = get_contextual_adapter()
        return self._local_adapter

    def _retrieve_top_k_candidates(
        self, db: Session, vector: list[float],
    ) -> list[dict]:
        """Выполняет векторный поиск top-K кандидатов среди узлов иерархии.

        Args:
            db: Сессия SQLAlchemy.
            vector: Векторное представление тега.

        Returns:
            Список словарей кандидатов с slug, name, path, similarity.
        """
        if not vector:
            return []

        try:
            rows = (
                db.query(
                    InterestHierarchyNode,
                    InterestHierarchyNode.embedding.cosine_distance(vector).label("distance"),
                )
                .filter(InterestHierarchyNode.embedding.isnot(None))
                .order_by("distance")
                .limit(self.top_k)
                .all()
            )
        except Exception as e:
            logger.exception(f"[stage1] Vector search failed: {e}")
            return []

        candidates = []
        for row in rows:
            node, distance = row
            similarity = 1.0 - float(distance) / 2.0
            candidates.append({
                "id": node.id,
                "slug": node.slug,
                "name": node.name,
                "path": node.path,
                "depth": node.depth,
                "similarity": round(float(similarity), 4),
            })

        return candidates

    def _cache_resolution(
        self, db: Session, raw_tag: str, slug: str,
        confidence: float, source: str = "llm_classifier",
        enriched_context: Optional[str] = None,
    ) -> None:
        """Кэширует результат разрешения тега в таблице DynamicAlias.

        Args:
            db: Сессия SQLAlchemy.
            raw_tag: Исходный тег.
            slug: Разрешённый слаг.
            confidence: Уверенность в разрешении.
            source: Источник разрешения.
            enriched_context: Обогащённый контекст.
        """
        try:
            tag_hash = hashlib.md5(raw_tag.encode()).hexdigest()
            existing = db.query(DynamicAlias).filter_by(raw_tag=raw_tag).first()
            if existing:
                existing.slug = slug
                existing.confidence = max(getattr(existing, "confidence", 0.0), float(confidence))
                existing.enriched_context = enriched_context or existing.enriched_context
                existing.source = source or existing.source
            else:
                db.add(DynamicAlias(
                    raw_tag=raw_tag, slug=slug, tag_hash=tag_hash,
                    confidence=float(confidence), enriched_context=enriched_context,
                    source=source,
                ))
            db.commit()
        except Exception as e:
            logger.exception(f"[cache] Failed: {e}")
            db.rollback()

    def _cache_unresolved(self, db: Session, raw_tag: str) -> None:
        """Помечает тег как неразрешимый в таблице DynamicAlias.

        Args:
            db: Сессия SQLAlchemy.
            raw_tag: Исходный тег, который не удалось разрешить.
        """
        try:
            tag_hash = hashlib.md5(raw_tag.encode()).hexdigest()
            existing = db.query(DynamicAlias).filter_by(raw_tag=raw_tag).first()
            if existing:
                existing.slug = UNRESOLVED_SENTINEL
                existing.confidence = 0.0
                existing.source = "unresolved"
            else:
                db.add(DynamicAlias(
                    raw_tag=raw_tag, slug=UNRESOLVED_SENTINEL, tag_hash=tag_hash,
                    confidence=0.0, source="unresolved",
                ))
            db.commit()
        except Exception as e:
            logger.exception(f"[cache] Failed to cache unresolved: {e}")
            db.rollback()

    def resolve_tag_to_slug(
        self, db: Session, raw_tag: str, fallback_to_enrichment: bool = True,
    ) -> Optional[str]:
        """Разрешает сырой тег пользователя в слаг иерархии интересов.

        Проверяет кэш, затем выполняет SBERT-поиск и опционально LLM-уточнение.

        Args:
            db: Сессия SQLAlchemy.
            raw_tag: Исходный тег пользователя.
            fallback_to_enrichment: Использовать LLM при неуверенном результате.

        Returns:
            Слаг иерархии или None, если разрешить не удалось.
        """
        if not raw_tag or not isinstance(raw_tag, str):
            return None

        norm_tag = raw_tag.strip().lower()
        if not norm_tag:
            return None

        cached = db.query(DynamicAlias).filter_by(raw_tag=norm_tag).first()
        if cached:
            if getattr(cached, "slug", None) == UNRESOLVED_SENTINEL:
                return None
            return cached.slug

        adapter = self._get_adapter()
        try:
            sbert = _get_sbert_model()
        except Exception as e:
            logger.exception(f"[resolve] SBERT unavailable: {e}")
            return None

        enriched = adapter.enrich_text(norm_tag).enriched
        try:
            vec = sbert.encode([enriched], convert_to_numpy=True)[0].tolist()
        except Exception as e:
            logger.exception(f"[resolve] SBERT encode failed: {e}")
            return None

        candidates = self._retrieve_top_k_candidates(db, vec)
        if not candidates:
            logger.warning(f"[resolve] No candidates found for '{norm_tag}'")
            self._cache_unresolved(db, norm_tag)
            return None

        logger.info(
            "[resolve] Stage 1 complete: %d candidates for '%s' (top: %s)",
            len(candidates), norm_tag, candidates[0]["slug"],
        )

        if fallback_to_enrichment:
            result = asyncio.run(self._llm_resolve_tag(norm_tag, candidates))
            if result is not None:
                if result["status"] == "matched":
                    slug = result["slug"]
                    confidence = result.get("confidence", 0.0)
                    source = f"llm_{result.get('provider', 'unknown')}"
                    self._cache_resolution(db, norm_tag, slug, confidence, source=source)
                    return slug

                if result["status"] == "create":
                    new_slug = result["suggested_slug"]
                    confidence = result.get("confidence", 0.0)
                    source = f"llm_{result.get('provider', 'unknown')}_new"
                    self._cache_resolution(db, norm_tag, new_slug, confidence, source=source)
                    logger.info(f"[resolve] LLM suggested NEW slug '{new_slug}' for '{norm_tag}'")
                    return new_slug

            logger.warning(f"[resolve] Stage 2 (LLM) failed for '%s', falling back to vector", norm_tag)

        best = candidates[0]
        slug = best["slug"]
        similarity = best["similarity"]
        if similarity >= 0.7:
            source = "vector_direct" if not fallback_to_enrichment else "vector_fallback"
            self._cache_resolution(db, norm_tag, slug, similarity, source=source)
            return slug

        self._cache_unresolved(db, norm_tag)
        return None

    async def _llm_resolve_tag(
        self, raw_tag: str, candidates: list[dict],
    ) -> Optional[dict]:
        """Запускает LLM-каскад для точного разрешения тега.

        Args:
            raw_tag: Исходный тег.
            candidates: Список кандидатов из векторного поиска.

        Returns:
            Результат классификации от LLM или None.
        """
        if not self._cascade:
            logger.warning("[stage2] No LLM cascade configured")
            return None

        return await self._cascade.classify(raw_tag, candidates)


_enricher: Optional[DynamicTagEnricher] = None


def get_tag_enricher() -> DynamicTagEnricher:
    """Возвращает единственный экземпляр DynamicTagEnricher (lazy singleton).

    Returns:
        Экземпляр DynamicTagEnricher.
    """
    global _enricher
    if _enricher is None:
        _enricher = DynamicTagEnricher()
    return _enricher
