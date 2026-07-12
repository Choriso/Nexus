"""
Robust Dynamic Tag Enricher (RAG) for Nexus

Goals implemented:
- Read all configuration from global `config` (no hard-coded URLs/models).
- SBERT is loaded once with local_files_only=True to avoid HuggingFace HEAD spam.
  If the model isn't available locally, it will be downloaded once (if permitted) and
  thereafter used locally. Network access is **not** attempted on every tag.
- Ollama requests use the correct endpoint (`{OLLAMA_BASE_URL.rstrip('/')}/api/generate`)
  and a strict JSON contract ({"model","prompt","stream":False}). Responses are
  sanitized before use.
- Strict fallback threshold: final similarity must be >= RAW_FALLBACK_MIN_SIMILARITY
  (set to 0.70). Otherwise the tag is considered unresolved and cached as such.

This module is intended to run during the WRITE phase (Celery). It must be resilient
and deterministic: never return random unrelated slugs for noisy input.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Optional, Tuple, Any

import aiohttp
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.ai_profiler.contextual_adapter import get_contextual_adapter
from config import config
from data.ai import DynamicAlias
from data.interest_hierarchy import InterestHierarchyNode

logger = logging.getLogger(__name__)

# Strict thresholds to avoid accidental noisy matches
HIGH_SIMILARITY_THRESHOLD = getattr(config, "HIGH_SIMILARITY_THRESHOLD", 0.85)
UNKNOWN_TAG_THRESHOLD = getattr(config, "UNKNOWN_TAG_THRESHOLD", 0.45)
RAW_FALLBACK_MIN_SIMILARITY = getattr(config, "RAW_FALLBACK_MIN_SIMILARITY", 0.70)

# Sentinel value to store unresolved mappings in DynamicAlias (DB slug column is non-null)
UNRESOLVED_SENTINEL = "__UNRESOLVED__"

# Global SBERT singleton and download flag
_sbert_model: Optional[SentenceTransformer] = None
_sbert_model_name: Optional[str] = None
_sbert_model_downloaded = False


def _get_sbert_model() -> SentenceTransformer:
    """Lazy-load the SBERT model once. Use local_files_only=True to avoid HF network spam.

    If model not present locally, attempt a single download (local_files_only=False) and
    keep that model in the cache for subsequent calls. Do NOT attempt network access on
    every call.
    """
    global _sbert_model, _sbert_model_name, _sbert_model_downloaded

    if _sbert_model is not None:
        return _sbert_model

    model_name = getattr(config, "SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    _sbert_model_name = model_name

    # Try local-only first to avoid network calls
    try:
        logger.debug(f"[sbert] Loading model locally: {model_name} (local_files_only=True)")
        _sbert_model = SentenceTransformer(model_name, model_kwargs={"local_files_only": True})
        logger.info("[sbert] Model loaded from local cache")
        return _sbert_model
    except Exception as e_local:
        logger.warning(f"[sbert] Local load failed for '{model_name}': {e_local}")

    # If local not available, attempt a single download if enabled in config
    allow_download = getattr(config, "SBERT_ALLOW_DOWNLOAD", True)
    if not allow_download:
        raise RuntimeError(
            f"SBERT model '{model_name}' not found locally and downloads are disabled (SBERT_ALLOW_DOWNLOAD=False)"
        )

    if not _sbert_model_downloaded:
        try:
            logger.info(f"[sbert] Attempting one-time download of model '{model_name}'")
            _sbert_model = SentenceTransformer(model_name, model_kwargs={"local_files_only": False})
            _sbert_model_downloaded = True
            logger.info("[sbert] Model downloaded and cached locally")
            return _sbert_model
        except Exception as e_dl:
            logger.exception(f"[sbert] Failed to download SBERT model '{model_name}': {e_dl}")
            raise
    else:
        # If we already attempted download but model still missing -> fail
        raise RuntimeError(f"SBERT model '{model_name}' unavailable after download attempt")


def _sanitize_text(text: str) -> Optional[str]:
    """Clean LLM / search snippets: remove code fences, excessive quotes, markup,
    control characters and short noise strings. Return None if result is not usable.
    """
    if not text or not isinstance(text, str):
        return None
    txt = text.strip()

    # Remove Markdown code fences and inline code
    txt = re.sub(r"```.*?```", "", txt, flags=re.S)
    txt = re.sub(r"`([^`]*)`", r"\1", txt)

    # Remove HTML tags
    txt = re.sub(r"<[^>]+>", "", txt)

    # Remove leading/trailing quotes and excessive punctuation
    txt = txt.strip('"\'')
    txt = re.sub(r"[\n\r]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt)

    # Discard if too short or obviously technical
    if len(txt) < 4:
        return None
    if re.search(r"^\W+$", txt):
        return None
    if any(token in txt.lower() for token in ("error", "traceback", "http", "404", "exception")):
        return None

    return txt.strip()


class DynamicTagEnricher:
    """Main enricher class. Minimal state; safe for repeated calls in Celery workers.

    Configuration is read from `config` module:
      - OLLAMA_ENABLED: bool
      - OLLAMA_BASE_URL: str
      - OLLAMA_MODEL: str
      - DUCKDUCKGO_ENABLED: bool
    """

    def __init__(self):
        self.ollama_enabled = getattr(config, "OLLAMA_ENABLED", False)
        self.ollama_url = getattr(config, "OLLAMA_BASE_URL", "").rstrip('/') if getattr(config, "OLLAMA_BASE_URL", None) else None
        self.ollama_model = getattr(config, "OLLAMA_MODEL", None)
        self.duckduckgo_enabled = getattr(config, "DUCKDUCKGO_ENABLED", True)

        # Local cache for small items to avoid repeated DB hits when possible
        self._local_adapter = None

    def _get_adapter(self):
        if self._local_adapter is None:
            self._local_adapter = get_contextual_adapter()
        return self._local_adapter

    def _find_closest_node(self, db: Session, vector: list[float]) -> Tuple[Optional[InterestHierarchyNode], float]:
        """Find the closest hierarchy node to a vector using pgvector cosine distance.

        Returns (node, similarity) where similarity is in [0.0, 1.0].
        """
        if not vector:
            return None, 0.0

        # Use SQL expression via SQLAlchemy; rely on existing pgvector column methods
        try:
            row = (
                db.query(InterestHierarchyNode, InterestHierarchyNode.embedding.cosine_distance(vector).label("distance"))
                .filter(InterestHierarchyNode.embedding.isnot(None))
                .order_by("distance")
                .first()
            )
        except Exception as e:
            logger.exception(f"[find_closest] DB vector search failed: {e}")
            return None, 0.0

        if not row:
            return None, 0.0

        node, distance = row
        # Pgvector cosine_distance returns in [-1,1] mapping; convert to similarity in [0,1]
        try:
            similarity = 1.0 - float(distance) / 2.0
        except Exception:
            similarity = 0.0

        return node, float(similarity)

    def resolve_tag_to_slug(self, db: Session, raw_tag: str, fallback_to_enrichment: bool = True) -> Optional[str]:
        """
        Resolve a raw tag to a canonical slug. Returns slug or None.

        Workflow:
        1. Check dynamic cache (DynamicAlias). If cached as unresolved sentinel, return None.
        2. Vector-search raw_tag using SBERT. If similarity >= HIGH_SIMILARITY_THRESHOLD => accept.
        3. If similarity >= RAW_FALLBACK_MIN_SIMILARITY => accept (low-confidence but acceptable).
        4. If similarity < RAW_FALLBACK_MIN_SIMILARITY and fallback_to_enrichment:
           - Enrich via Ollama or DuckDuckGo
           - Re-vectorize enriched text and accept only if similarity >= RAW_FALLBACK_MIN_SIMILARITY
        5. If final similarity < RAW_FALLBACK_MIN_SIMILARITY -> cache UNRESOLVED and return None
        """
        if not raw_tag or not isinstance(raw_tag, str):
            return None

        norm_tag = raw_tag.strip().lower()
        if not norm_tag:
            return None

        # 1) Check cache
        try:
            cached = db.query(DynamicAlias).filter_by(raw_tag=norm_tag).first()
            if cached:
                # Treat unresolved sentinel as None
                if getattr(cached, "slug", None) == UNRESOLVED_SENTINEL:
                    logger.debug(f"[resolve] Cached UNRESOLVED for '{norm_tag}'")
                    return None
                logger.debug(f"[resolve] Cache hit for '{norm_tag}' -> '{cached.slug}'")
                return cached.slug
        except Exception as e:
            logger.warning(f"[resolve] DynamicAlias lookup failed: {e}")

        # 2) Vector search raw tag
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
            logger.exception(f"[resolve] SBERT encode failed for '{norm_tag}': {e}")
            return None

        node, similarity = self._find_closest_node(db, vec)
        logger.debug(f"[resolve] '{norm_tag}' -> initial similarity={similarity:.3f} (node={(node.slug if node else None)})")

        # Accept if high similarity
        if node and similarity >= HIGH_SIMILARITY_THRESHOLD:
            self._cache_resolution(db, norm_tag, node.slug, similarity, source="vector_direct")
            return node.slug

        # If similarity already acceptable (>= RAW_FALLBACK_MIN_SIMILARITY), accept but mark low confidence
        if node and similarity >= RAW_FALLBACK_MIN_SIMILARITY:
            self._cache_resolution(db, norm_tag, node.slug, similarity, source="vector_weak")
            return node.slug

        # 3) Fallback enrichment if allowed
        if fallback_to_enrichment:
            logger.info(f"[resolve] '{norm_tag}' below fallback threshold ({similarity:.3f}), attempting enrichment")
            enriched_context = asyncio.run(self._enrich_unknown_tag(norm_tag))
            if not enriched_context:
                logger.info(f"[resolve] Enrichment provided no useful context for '{norm_tag}'")
                self._cache_unresolved(db, norm_tag)
                return None

            # Re-encode enriched context
            enriched_for_encode = adapter.enrich_text(enriched_context).enriched
            try:
                vec2 = sbert.encode([enriched_for_encode], convert_to_numpy=True)[0].tolist()
            except Exception as e:
                logger.exception(f"[resolve] SBERT encode failed for enriched text of '{norm_tag}': {e}")
                self._cache_unresolved(db, norm_tag)
                return None

            node2, similarity2 = self._find_closest_node(db, vec2)
            logger.debug(f"[resolve] '{norm_tag}' -> enriched similarity={similarity2:.3f} (node={(node2.slug if node2 else None)})")

            if node2 and similarity2 >= RAW_FALLBACK_MIN_SIMILARITY:
                # Good enriched match
                self._cache_resolution(db, norm_tag, node2.slug, similarity2, enriched_context=enriched_context, source="enriched")
                return node2.slug

            # If enriched result still too low -> unresolved
            logger.info(f"[resolve] Enriched similarity {similarity2:.3f} below RAW_FALLBACK_MIN_SIMILARITY for '{norm_tag}' - marking unresolved")
            self._cache_unresolved(db, norm_tag, enriched_context)
            return None

        # No fallback allowed -> mark unresolved
        self._cache_unresolved(db, norm_tag)
        return None

    async def _enrich_unknown_tag(self, raw_tag: str, timeout: int = 6) -> Optional[str]:
        """Attempt to enrich unknown tag via Ollama (preferred) then DuckDuckGo.

        Returns sanitized text or None.
        """
        results = []

        # Ollama first (if enabled)
        if self.ollama_enabled and self.ollama_url and self.ollama_model:
            try:
                got = await self._enrich_via_ollama(raw_tag, timeout)
                if got:
                    sanitized = _sanitize_text(got)
                    if sanitized:
                        return sanitized
            except Exception as e:
                logger.warning(f"[enrich] Ollama enrichment failed for '{raw_tag}': {e}")

        # DuckDuckGo fallback
        if self.duckduckgo_enabled:
            try:
                got = await self._enrich_via_duckduckgo(raw_tag, timeout)
                if got:
                    sanitized = _sanitize_text(got)
                    if sanitized:
                        return sanitized
            except Exception as e:
                logger.warning(f"[enrich] DuckDuckGo enrichment failed for '{raw_tag}': {e}")

        return None

    async def _enrich_via_ollama(self, raw_tag: str, timeout: int = 6) -> Optional[str]:
        """Call Ollama /api/generate with strict contract.

        Body: {"model": <model>, "prompt": <prompt>, "stream": False}
        Expectation: JSON with some string content; sanitize before returning.
        """
        if not self.ollama_url or not self.ollama_model:
            return None

        url = f"{self.ollama_url}/api/generate"
        prompt = f"Provide a short one-sentence description of the term: '{raw_tag}'." \
                 " If ambiguous, provide concise clarifying context."

        payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    text_resp = await resp.text()
                    if resp.status == 404:
                        logger.warning(f"[ollama] 404 from {url} - check OLLAMA_BASE_URL")
                        return None
                    if resp.status >= 400:
                        logger.warning(f"[ollama] HTTP {resp.status} for '{raw_tag}': {text_resp[:200]}")
                        return None

                    try:
                        data = await resp.json()
                    except Exception:
                        # If response is not valid JSON, try plain text
                        data = {"response": text_resp}

                    # Extract plausible text from known keys
                    candidates = []
                    if isinstance(data, dict):
                        for key in ("response", "output", "generated", "text", "result"):
                            if key in data and data[key]:
                                candidates.append(data[key])
                        # Also check for nested generated arrays
                        if not candidates and "generated" in data and isinstance(data["generated"], list):
                            for item in data["generated"]:
                                if isinstance(item, dict) and item.get("content"):
                                    candidates.append(item.get("content"))
                                elif isinstance(item, str):
                                    candidates.append(item)

                    # Flatten and pick first non-empty string
                    for c in candidates:
                        if isinstance(c, str) and c.strip():
                            return c.strip()
                        if isinstance(c, list):
                            for e in c:
                                if isinstance(e, str) and e.strip():
                                    return e.strip()

                    # As last resort, use raw text response
                    return text_resp.strip() if text_resp and text_resp.strip() else None
        except asyncio.TimeoutError:
            logger.warning(f"[ollama] Timeout enriching '{raw_tag}'")
            return None
        except Exception as e:
            logger.exception(f"[ollama] Error enriching '{raw_tag}': {e}")
            return None

    async def _enrich_via_duckduckgo(self, raw_tag: str, timeout: int = 5) -> Optional[str]:
        """Query DuckDuckGo instant answer API for a short abstract.

        URL: https://api.duckduckgo.com/?q=<raw_tag>&format=json
        """
        try:
            url = "https://api.duckduckgo.com/"
            params = {"q": raw_tag, "format": "json", "t": "nexus_enrichment"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status != 200:
                        logger.warning(f"[ddg] HTTP {resp.status} for '{raw_tag}'")
                        return None
                    data = await resp.json()
                    # Prefer AbstractText
                    abstract = data.get("AbstractText")
                    if abstract:
                        return abstract
                    # Else fall back to first Result text
                    results = data.get("Results", [])
                    if results and isinstance(results, list):
                        for r in results:
                            if isinstance(r, dict) and r.get("Text"):
                                return r.get("Text")
        except asyncio.TimeoutError:
            logger.warning(f"[ddg] Timeout for '{raw_tag}'")
        except Exception as e:
            logger.exception(f"[ddg] Error for '{raw_tag}': {e}")
        return None

    def _cache_resolution(self, db: Session, raw_tag: str, slug: str, similarity: float, enriched_context: Optional[str] = None, source: Optional[str] = None) -> None:
        """Persist a successful resolution into DynamicAlias.

        Stores: raw_tag, slug, confidence, enriched_context, source
        """
        try:
            tag_hash = hashlib.md5(raw_tag.encode()).hexdigest()
            existing = db.query(DynamicAlias).filter_by(raw_tag=raw_tag).first()
            if existing:
                existing.slug = slug
                existing.confidence = max(getattr(existing, "confidence", 0.0), float(similarity))
                existing.enriched_context = enriched_context or existing.enriched_context
                existing.source = source or existing.source
            else:
                db.add(DynamicAlias(raw_tag=raw_tag, slug=slug, tag_hash=tag_hash, confidence=float(similarity), enriched_context=enriched_context, source=source))
            db.commit()
            logger.debug(f"[cache] Cached '{raw_tag}' -> '{slug}' (sim={similarity:.3f})")
        except Exception as e:
            logger.exception(f"[cache] Could not cache resolution for '{raw_tag}': {e}")
            db.rollback()

    def _cache_unresolved(self, db: Session, raw_tag: str, enriched_context: Optional[str] = None) -> None:
        """Cache unresolved tag using UNRESOLVED_SENTINEL so we don't retry noisy tags repeatedly."""
        try:
            tag_hash = hashlib.md5(raw_tag.encode()).hexdigest()
            existing = db.query(DynamicAlias).filter_by(raw_tag=raw_tag).first()
            if existing:
                existing.slug = UNRESOLVED_SENTINEL
                existing.confidence = 0.0
                existing.enriched_context = enriched_context or existing.enriched_context
                existing.source = existing.source or "unresolved"
            else:
                db.add(DynamicAlias(raw_tag=raw_tag, slug=UNRESOLVED_SENTINEL, tag_hash=tag_hash, confidence=0.0, enriched_context=enriched_context, source="unresolved"))
            db.commit()
            logger.debug(f"[cache] Marked '{raw_tag}' as UNRESOLVED in DynamicAlias")
        except Exception as e:
            logger.exception(f"[cache] Could not cache unresolved for '{raw_tag}': {e}")
            db.rollback()


# Singleton enricher
_enricher: Optional[DynamicTagEnricher] = None


def get_tag_enricher() -> DynamicTagEnricher:
    global _enricher
    if _enricher is None:
        _enricher = DynamicTagEnricher()
    return _enricher
