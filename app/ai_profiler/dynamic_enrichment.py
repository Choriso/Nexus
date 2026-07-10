"""
Dynamic tag enrichment via RAG (Retrieval-Augmented Generation).

When a raw tag from user text doesn't map to known interests (low vector similarity),
this module enriches the tag's context using external sources (DuckDuckGo, Ollama)
and maps it to the hierarchy. Results are cached in DynamicAliases table.

This runs during the WRITE phase (Celery workers), NOT in sync request cycles.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Optional

import aiohttp
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.ai_profiler.contextual_adapter import get_contextual_adapter
from config import config
from data.ai import DynamicAlias, InterestHierarchyNode

logger = logging.getLogger(__name__)

# Thresholds for similarity detection
HIGH_SIMILARITY_THRESHOLD = 0.75
UNKNOWN_TAG_THRESHOLD = 0.45


class DynamicTagEnricher:
    """
    Enriches unknown tags via web search or LLM, maps to hierarchy, caches results.
    
    Design: Stateless service, safe for concurrent use. All I/O is async.
    """
    
    def __init__(
        self,
        sbert_model: Optional[SentenceTransformer] = None,
        ollama_url: str | None = None,
        duckduckgo_enabled: bool = True,
    ):
        """
        Args:
            sbert_model: Pre-loaded SentenceTransformer (or None to load on demand)
            ollama_url: URL to Ollama service (e.g., "http://localhost:11434")
            duckduckgo_enabled: Use DuckDuckGo for web search enrichment
        """
        self.sbert_model = sbert_model
        self.ollama_url = ollama_url or getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
        self.duckduckgo_enabled = duckduckgo_enabled

    def _get_sbert(self) -> SentenceTransformer:
        """Lazy load SBERT model."""
        if self.sbert_model is None:
            adapter = get_contextual_adapter()
            self.sbert_model = adapter.sbert_model
        return self.sbert_model

    def _find_closest_node(
        self, db: Session, vector: list[float], exclude_node_id: Optional[int] = None
    ) -> tuple[Optional[InterestHierarchyNode], float]:
        """
        Find the closest hierarchy node to a vector using pgvector distance.
        
        Returns: (node, similarity_score) where similarity = 1.0 - normalized_distance
        """
        from sqlalchemy import func, text
        
        if not vector:
            return None, 0.0
        
        # Use pgvector <-> operator for cosine distance
        query = db.query(
            InterestHierarchyNode,
            (1.0 - func.cast(InterestHierarchyNode.embedding.cosine_distance(vector), float) / 2.0).label("similarity")
        ).filter(
            InterestHierarchyNode.embedding.isnot(None)
        )
        
        if exclude_node_id:
            query = query.filter(InterestHierarchyNode.id != exclude_node_id)
        
        result = query.order_by("similarity").first()
        if result:
            node, similarity = result
            return node, float(similarity)
        
        return None, 0.0

    def resolve_tag_to_slug(
        self,
        db: Session,
        raw_tag: str,
        fallback_to_enrichment: bool = True,
    ) -> Optional[str]:
        """
        Resolve a raw tag to a hierarchy slug.
        
        Flow:
        1. Check cache (DynamicAlias)
        2. Vector search in hierarchy
        3. If similarity too low AND fallback_to_enrichment:
           - Enrich tag context
           - Re-search with enriched text
           - Cache result
        
        Args:
            db: SQLAlchemy session
            raw_tag: Raw text tag (e.g., "cs2", "я люблю музыку")
            fallback_to_enrichment: Try enrichment if vector search fails
        
        Returns:
            slug (str) or None if resolution failed
        """
        if not raw_tag or not isinstance(raw_tag, str):
            return None
        
        raw_tag = raw_tag.lower().strip()
        
        # 1. Check dynamic cache
        cached = db.query(DynamicAlias).filter_by(raw_tag=raw_tag).first()
        if cached:
            logger.debug(f"[resolve_tag] Cache hit for '{raw_tag}' -> '{cached.slug}'")
            return cached.slug
        
        # 2. Vector search for raw tag
        sbert = self._get_sbert()
        adapter = get_contextual_adapter()
        
        enriched_raw = adapter.enrich_text(raw_tag).enriched
        raw_vector = sbert.encode([enriched_raw], convert_to_numpy=True)[0].tolist()
        
        closest_node, similarity = self._find_closest_node(db, raw_vector)
        logger.debug(f"[resolve_tag] '{raw_tag}' -> similarity={similarity:.3f}")
        
        if similarity >= HIGH_SIMILARITY_THRESHOLD and closest_node:
            logger.info(f"[resolve_tag] Direct match: '{raw_tag}' -> '{closest_node.slug}' (sim={similarity:.3f})")
            self._cache_resolution(db, raw_tag, closest_node.slug)
            return closest_node.slug
        
        # 3. Fallback: enrich unknown tag
        if fallback_to_enrichment and similarity < UNKNOWN_TAG_THRESHOLD:
            logger.info(f"[resolve_tag] Unknown tag '{raw_tag}' (sim={similarity:.3f}), attempting enrichment...")
            enriched_context = asyncio.run(self._enrich_unknown_tag(raw_tag))
            
            if enriched_context:
                logger.debug(f"[resolve_tag] Enriched '{raw_tag}': {enriched_context[:100]}...")
                
                # Re-encode enriched text
                enriched_full = adapter.enrich_text(enriched_context).enriched
                enriched_vector = sbert.encode([enriched_full], convert_to_numpy=True)[0].tolist()
                
                closest_node, similarity = self._find_closest_node(db, enriched_vector)
                
                if similarity >= HIGH_SIMILARITY_THRESHOLD and closest_node:
                    logger.info(
                        f"[resolve_tag] Enrichment success: '{raw_tag}' -> '{closest_node.slug}' (sim={similarity:.3f})"
                    )
                    self._cache_resolution(db, raw_tag, closest_node.slug)
                    return closest_node.slug
        
        logger.warning(f"[resolve_tag] Resolution failed for '{raw_tag}'")
        return None

    async def _enrich_unknown_tag(self, raw_tag: str, timeout: int = 5) -> Optional[str]:
        """
        Enrich unknown tag context via web search or LLM.
        
        Strategy:
        1. Try Ollama LLM (if configured)
        2. Fall back to DuckDuckGo web search
        
        Returns: enriched context (string) or None
        """
        if getattr(config, "OLLAMA_ENABLED", False):
            result = await self._enrich_via_ollama(raw_tag, timeout)
            if result:
                return result
        
        if self.duckduckgo_enabled:
            result = await self._enrich_via_duckduckgo(raw_tag, timeout)
            if result:
                return result
        
        return None

    async def _enrich_via_ollama(self, raw_tag: str, timeout: int = 5) -> Optional[str]:
        """
        Query local Ollama instance for tag definition/context.
        
        Prompt: "In one sentence, what is {raw_tag}?"
        """
        try:
            prompt = f"Provide a brief one-sentence definition for the term '{raw_tag}'. Answer in English or Russian."
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": getattr(config, "OLLAMA_MODEL", "mistral"),
                        "prompt": prompt,
                        "stream": False,
                        "num_predict": 50,
                    },
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"[enrich_ollama] HTTP {resp.status} for '{raw_tag}'")
                        return None
                    
                    data = await resp.json()
                    response_text = data.get("response", "").strip()
                    
                    if response_text:
                        logger.debug(f"[enrich_ollama] Got response for '{raw_tag}': {response_text[:80]}...")
                        return response_text
        except asyncio.TimeoutError:
            logger.warning(f"[enrich_ollama] Timeout for '{raw_tag}'")
        except Exception as e:
            logger.warning(f"[enrich_ollama] Error: {e}")
        
        return None

    async def _enrich_via_duckduckgo(self, raw_tag: str, timeout: int = 5) -> Optional[str]:
        """
        Query DuckDuckGo for tag definition/context.
        
        Parses instant answer or first search snippet.
        """
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": raw_tag,
                "format": "json",
                "t": "nexus_enrichment",
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"[enrich_duckduckgo] HTTP {resp.status} for '{raw_tag}'")
                        return None
                    
                    data = await resp.json()
                    
                    # Try instant answer first
                    if data.get("AbstractText"):
                        return data["AbstractText"]
                    
                    # Fall back to first result snippet
                    results = data.get("Results", [])
                    if results and results[0].get("Text"):
                        return results[0]["Text"]
        except asyncio.TimeoutError:
            logger.warning(f"[enrich_duckduckgo] Timeout for '{raw_tag}'")
        except Exception as e:
            logger.warning(f"[enrich_duckduckgo] Error: {e}")
        
        return None

    def _cache_resolution(self, db: Session, raw_tag: str, slug: str) -> None:
        """Cache raw_tag → slug mapping in DynamicAliases table."""
        try:
            tag_hash = hashlib.md5(raw_tag.encode()).hexdigest()
            
            # Check if already cached
            existing = db.query(DynamicAlias).filter_by(raw_tag=raw_tag).first()
            if existing:
                existing.slug = slug
                existing.confidence = 0.95  # High confidence after enrichment
            else:
                alias = DynamicAlias(
                    raw_tag=raw_tag,
                    slug=slug,
                    confidence=0.95,
                    tag_hash=tag_hash,
                )
                db.add(alias)
            
            db.commit()
            logger.debug(f"[cache_resolution] Cached '{raw_tag}' -> '{slug}'")
        except Exception as e:
            logger.warning(f"[cache_resolution] Error: {e}")
            db.rollback()


# Global singleton for reuse
_enricher_instance: Optional[DynamicTagEnricher] = None


def get_tag_enricher() -> DynamicTagEnricher:
    """Get or create global enricher instance."""
    global _enricher_instance
    if _enricher_instance is None:
        _enricher_instance = DynamicTagEnricher(
            ollama_url=getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434"),
            duckduckgo_enabled=getattr(config, "DUCKDUCKGO_ENABLED", True),
        )
    return _enricher_instance
