from __future__ import annotations

import asyncio
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy.orm import Session

from app.ai_profiler.contextual_adapter import get_contextual_adapter
from app.ai_profiler.providers import FailoverCascade, build_cascade
from app.ai_profiler.semantic_ontology import SEMANTIC_ONTOLOGY
from config import config
from data.ai import DynamicAlias
from data.interest_hierarchy import InterestHierarchyNode

logger = logging.getLogger(__name__)

UNRESOLVED_SENTINEL = "__UNRESOLVED__"
LOCAL_TAG_THRESHOLD = 0.65
HYBRID_RRF_K = 60

_sbert_model: Optional[SentenceTransformer] = None
_sbert_model_name: Optional[str] = None
_sbert_model_downloaded = False
_cascade: Optional[FailoverCascade] = None

_tfidf_vectorizer: Optional[TfidfVectorizer] = None
_tfidf_slugs: Optional[list[str]] = None
_tfidf_matrix: Optional[np.ndarray] = None


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
        self._alias_to_slug_cache = None

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

    def _build_tfidf(self, db: Session) -> None:
        global _tfidf_vectorizer, _tfidf_slugs, _tfidf_matrix
        if _tfidf_vectorizer is not None:
            return
        path = Path(config.TFIDF_VECTORIZER_PATH)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                _tfidf_vectorizer = data["vectorizer"]
                _tfidf_slugs = data["slugs"]
                _tfidf_matrix = data["matrix"]
                logger.info("[tfidf] Loaded TF-IDF index for %d nodes from %s", len(_tfidf_slugs), path)
                return
            except Exception as e:
                logger.warning("[tfidf] Failed to load from disk: %s", e)
        nodes = db.query(InterestHierarchyNode).all()
        if not nodes:
            logger.warning("[tfidf] No hierarchy nodes found, skipping TF-IDF init")
            return
        docs: list[str] = []
        slugs: list[str] = []
        for n in nodes:
            parts = [n.name, n.slug]
            onto = SEMANTIC_ONTOLOGY.get(n.slug)
            if onto:
                enriched = onto.get("enriched_text", "")
                if enriched:
                    parts.append(enriched)
                aliases = onto.get("aliases", [])
                if aliases:
                    parts.extend(aliases)
            docs.append(" ".join(parts))
            slugs.append(n.slug)
        vec = TfidfVectorizer(max_features=5000, stop_words=None, lowercase=True, analyzer="word")
        matrix = vec.fit_transform(docs).toarray()
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        _tfidf_vectorizer = vec
        _tfidf_slugs = slugs
        _tfidf_matrix = matrix
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": vec, "slugs": slugs, "matrix": matrix}, f)
        logger.info("[tfidf] Built TF-IDF index for %d nodes", len(nodes))

    def _tfidf_search(self, norm_tag: str, top_k: int = 20) -> list[dict]:
        global _tfidf_vectorizer, _tfidf_slugs, _tfidf_matrix
        if _tfidf_vectorizer is None or _tfidf_slugs is None or _tfidf_matrix is None:
            return []
        try:
            query_vec = _tfidf_vectorizer.transform([norm_tag]).toarray()
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return []
            query_vec = query_vec / query_norm
            sims = _tfidf_matrix.dot(query_vec.T).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in top_indices:
                if sims[idx] > 0:
                    results.append({
                        "slug": _tfidf_slugs[idx],
                        "similarity": round(float(sims[idx]), 4),
                    })
            return results
        except Exception as e:
            logger.warning("[tfidf] Search failed: %s", e)
            return []

    def _hybrid_search(
        self, db: Session, norm_tag: str, enriched: Optional[str] = None,
    ) -> Optional[str]:
        self._build_tfidf(db)
        sbert = None
        try:
            sbert = _get_sbert_model()
        except Exception as e:
            logger.warning("[hybrid] SBERT unavailable: %s", e)

        vector_candidates: list[dict] = []
        if sbert:
            try:
                if not enriched:
                    adapter = self._get_adapter()
                    enriched = adapter.enrich_text(norm_tag).enriched
                vec = sbert.encode([enriched], convert_to_numpy=True)[0].tolist()
                if vec:
                    vector_candidates = self._retrieve_top_k_candidates(db, vec)
            except Exception as e:
                logger.warning("[hybrid] SBERT encode failed: %s", e)

        tfidf_candidates: list[dict] = self._tfidf_search(norm_tag)
        if not vector_candidates and not tfidf_candidates:
            logger.info("[hybrid] No candidates from either method for '%s'", norm_tag)
            return None

        combined: dict[str, dict] = {}
        for candidates, source in [(vector_candidates, "vector"), (tfidf_candidates, "tfidf")]:
            for rank, c in enumerate(candidates):
                key = c["slug"]
                if key not in combined:
                    combined[key] = {"slug": key, "rrf": 0.0, "vector_sim": 0.0, "tfidf_sim": 0.0}
                combined[key]["rrf"] += 1.0 / (rank + HYBRID_RRF_K)
                if source == "vector":
                    combined[key]["vector_sim"] = c.get("similarity", 0.0)
                    combined[key]["sbert_sim"] = c.get("similarity", 0.0)
                else:
                    combined[key]["tfidf_sim"] = c.get("similarity", 0.0)

        merged = sorted(combined.values(), key=lambda x: x["rrf"], reverse=True)

        best = merged[0]
        best_sim = max(best.get("vector_sim", 0.0), best.get("tfidf_sim", 0.0))
        logger.info(
            "[hybrid] Top for '%s': slug=%s rrf=%.4f vec=%.3f tfidf=%.3f",
            norm_tag, best["slug"], best["rrf"], best.get("vector_sim", 0.0), best.get("tfidf_sim", 0.0),
        )

        if best_sim >= LOCAL_TAG_THRESHOLD:
            logger.info("[hybrid] Accept '%s' for '%s' (sim=%.3f >= %.2f)", best["slug"], norm_tag, best_sim, LOCAL_TAG_THRESHOLD)
            return best["slug"]

        return None

    def resolve_tag_to_slug(
        self, db: Session, raw_tag: str, fallback_to_enrichment: bool = True,
        force: bool = False,
    ) -> Optional[str]:
        if not raw_tag or not isinstance(raw_tag, str):
            return None

        norm_tag = raw_tag.strip().lower()
        if not norm_tag:
            return None

        logger.info("[resolve_tag_to_slug] Entry: norm_tag='%s' force=%s", norm_tag, force)

        cached = db.query(DynamicAlias).filter_by(raw_tag=norm_tag).first()
        if cached:
            if getattr(cached, "slug", None) == UNRESOLVED_SENTINEL:
                if not force:
                    logger.info("[resolve_tag_to_slug] '%s' cached as UNRESOLVED, returning None", norm_tag)
                    return None
                logger.info("[resolve_tag_to_slug] '%s' cached as UNRESOLVED, force=True, deleting cache", norm_tag)
                db.delete(cached)
                db.commit()
            else:
                logger.info("[resolve_tag_to_slug] '%s' cached as '%s'", norm_tag, cached.slug)
                return cached.slug

        sbert = None
        try:
            sbert = _get_sbert_model()
        except Exception as e:
            logger.warning(f"[resolve] SBERT unavailable: {e}")

        enriched: Optional[str] = None
        vec: Optional[list[float]] = None
        candidates: list[dict] = []

        if sbert:
            try:
                adapter = self._get_adapter()
                enriched = adapter.enrich_text(norm_tag).enriched
                vec = sbert.encode([enriched], convert_to_numpy=True)[0].tolist()
                if vec:
                    candidates = self._retrieve_top_k_candidates(db, vec)
                    if candidates:
                        logger.info(
                            "[resolve] Vector candidates for '%s': top=%s (sim=%.3f), count=%d",
                            norm_tag, candidates[0]["slug"], candidates[0]["similarity"], len(candidates),
                        )
            except Exception as e:
                logger.warning(f"[resolve] SBERT pipeline failed: {e}")

        if candidates and fallback_to_enrichment:
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
            logger.warning(f"[resolve] LLM cascade failed for '%s'", norm_tag)

        hybrid_slug = self._hybrid_search(db, norm_tag, enriched=enriched)
        if hybrid_slug:
            self._cache_resolution(db, norm_tag, hybrid_slug, LOCAL_TAG_THRESHOLD, source="hybrid")
            return hybrid_slug

        keyword = self._keyword_match(db, norm_tag)
        if keyword:
            logger.info(f"[resolve] Keyword match '{keyword}' for '{norm_tag}'")
            self._cache_resolution(db, norm_tag, keyword, 0.5, source="keyword")
            return keyword

        logger.info("[resolve_tag_to_slug] All methods failed for '%s', caching as UNRESOLVED", norm_tag)
        self._cache_unresolved(db, norm_tag)
        return None

    def _keyword_match(self, db: Session, norm_tag: str) -> Optional[str]:
        from sqlalchemy import or_
        from data.interest_hierarchy import InterestHierarchyNode
        from app.ai_profiler.semantic_ontology import SEMANTIC_ONTOLOGY

        if self._alias_to_slug_cache is None:
            self._alias_to_slug_cache = {}
            for slug, entry in SEMANTIC_ONTOLOGY.items():
                for alias in entry.get("aliases", []):
                    self._alias_to_slug_cache[alias.lower().strip()] = slug

        def _resolve(s: str) -> Optional[str]:
            n = db.query(InterestHierarchyNode).filter_by(slug=s).first()
            return s if n else None

        slug = self._alias_to_slug_cache.get(norm_tag)
        if slug:
            result = _resolve(slug)
            if result:
                logger.info("[keyword_match] Exact alias '%s' -> '%s'", norm_tag, slug)
                return result

        best: Optional[str] = None
        best_len = 0
        for alias, mapped_slug in self._alias_to_slug_cache.items():
            if alias in norm_tag and len(alias) > best_len:
                if _resolve(mapped_slug):
                    best = mapped_slug
                    best_len = len(alias)
        if best:
            logger.info("[keyword_match] Partial alias '%s' contains alias -> '%s'", norm_tag, best)
            return best

        try:
            fuzzy = db.query(InterestHierarchyNode).filter(
                or_(
                    InterestHierarchyNode.slug.ilike(f"%{norm_tag}%"),
                    InterestHierarchyNode.name.ilike(f"%{norm_tag}%"),
                )
            ).first()
            if fuzzy:
                return fuzzy.slug
        except Exception:
            pass
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
