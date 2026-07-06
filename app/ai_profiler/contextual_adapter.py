"""
ContextualAdapter — слой нормализации и обогащения контекста перед SBERT.encode().

Перехватывает сырые теги, интересы и текст узлов графа знаний, раскрывает
сленг через онтологический словарь и таксономию, формируя семантически
богатые строки для векторного сравнения.

Использование:
    adapter = ContextualAdapter()
    enriched = adapter.enrich_text("катнуть в кэсочку")
    # → "компьютерные и видеоигры ... Counter-Strike 2 ..."

    texts = adapter.prepare_for_encoding(["Flask API", "бэкенд на Python"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from .semantic_ontology import SEMANTIC_ONTOLOGY
from .taxonomy import INTEREST_TAXONOMY
from config import config
_contextual_adapter_instance = None
_SHARED_SBERT_MODEL = None

@dataclass
class EnrichmentResult:
    """Результат обогащения одного фрагмента текста."""

    original: str
    enriched: str
    matched_concepts: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    subcategories: list[str] = field(default_factory=list)


class ContextualAdapter:
    """
    Контекстуальный адаптер Nexus: превращает сленг и аббревиатуры
    в семантически насыщенный текст, понятный SBERT.

    Архитектура (три уровня):
        1. Онтологический словарь (SEMANTIC_ONTOLOGY) — точные совпадения
           сленга/аббревиатур с развёрнутыми описаниями.
        2. Таксономия (INTEREST_TAXONOMY) — привязка к подкатегориям и
           добавление якорных фраз при частичном совпадении.
        3. Эвристики — раскрытие CamelCase, ALL_CAPS аббревиатур, транслита.

    Все методы идемпотентны и безопасны для пустого ввода.
    """

    _WORD_RE = re.compile(r"[а-яёa-z0-9]+(?:[/\-][а-яёa-z0-9]+)?", re.IGNORECASE)

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._alias_index = self._build_alias_index()
        self._taxonomy_index = self._build_taxonomy_index()
        global _SHARED_SBERT_MODEL
        if _SHARED_SBERT_MODEL is None:
            from sentence_transformers import SentenceTransformer
            # Берем модель строго из конфига проекта Nexus
            model_name = getattr(config, 'EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
            _SHARED_SBERT_MODEL = SentenceTransformer(model_name)

        self.sbert_model = _SHARED_SBERT_MODEL

    @staticmethod
    def _build_alias_index() -> list[tuple[str, str, dict]]:
        """
        Строит плоский индекс (alias, canonical_key, entry), отсортированный
        по длине alias (длинные совпадения первыми — greedy longest match).
        """
        entries: list[tuple[str, str, dict]] = []
        for key, entry in SEMANTIC_ONTOLOGY.items():
            for alias in entry.get("aliases", []):
                entries.append((alias.lower().strip(), key, entry))
        entries.sort(key=lambda x: len(x[0]), reverse=True)
        return entries

    @staticmethod
    def _build_taxonomy_index() -> list[tuple[str, str, str, list[str]]]:
        """
        Индекс таксономии: (subcategory_lower, global_cat, subcategory, anchors).
        """
        index: list[tuple[str, str, str, list[str]]] = []
        for global_cat, subcats in INTEREST_TAXONOMY.items():
            for subcategory, anchors in subcats.items():
                index.append((subcategory.lower(), global_cat, subcategory, anchors))
        return index

    def enrich_text(self, text: str) -> EnrichmentResult:
        """
        Обогащает произвольный текст для последующего SBERT.encode().

        Алгоритм:
            1. Нормализация регистра для поиска.
            2. Longest-match поиск алиасов в онтологии.
            3. Дополнение контекстом из таксономии (если совпала подкатегория).
            4. Сборка итоговой строки: оригинал + развёрнутые концепты.

        Args:
            text: Сырой текст (тег, заголовок узла, поисковый запрос).

        Returns:
            EnrichmentResult с полем enriched — строка для encode().
        """
        if not self.enabled or not text or not text.strip():
            return EnrichmentResult(original=text or "", enriched=text or "")

        normalized = text.strip()
        lower = normalized.lower()

        matched_concepts: list[str] = []
        categories: list[str] = []
        subcategories: list[str] = []
        enrichments: list[str] = []
        consumed_spans: list[tuple[int, int]] = []

        # Уровень 1: онтологический словарь (longest match)
        for alias, concept_key, entry in self._alias_index:
            start = 0
            while True:
                pos = lower.find(alias, start)
                if pos == -1:
                    break
                end = pos + len(alias)
                if self._span_overlaps(consumed_spans, pos, end):
                    start = pos + 1
                    continue
                consumed_spans.append((pos, end))
                matched_concepts.append(concept_key)
                enrichments.append(entry["enriched_text"])
                if entry.get("category") and entry["category"] not in categories:
                    categories.append(entry["category"])
                if entry.get("subcategory") and entry["subcategory"] not in subcategories:
                    subcategories.append(entry["subcategory"])
                start = end

        # Уровень 2: таксономия — частичное совпадение с подкатегорией или якорем
        for sub_lower, global_cat, subcategory, anchors in self._taxonomy_index:
            if sub_lower in lower and subcategory not in subcategories:
                subcategories.append(subcategory)
                if global_cat not in categories:
                    categories.append(global_cat)
                enrichments.append(f"{subcategory}: {', '.join(anchors[:4])}")
            else:
                for anchor in anchors[:6]:
                    if len(anchor) >= 4 and anchor.lower() in lower:
                        if subcategory not in subcategories:
                            subcategories.append(subcategory)
                            enrichments.append(
                                f"{subcategory} ({global_cat}): {anchor}"
                            )
                        break

        # Уровень 3: эвристики для непокрытых токенов
        tokens = self._WORD_RE.findall(lower)
        for token in tokens:
            if len(token) <= 2:
                continue
            if token.isupper() and len(token) <= 6:
                enrichments.append(f"аббревиатура {token}")
            elif self._is_latin_only(token) and len(token) >= 3:
                enrichments.append(f"англицизм {token}, технический термин")

        # Сборка: оригинал сохраняется + семантическое обогащение
        unique_enrichments = list(dict.fromkeys(enrichments))
        if unique_enrichments:
            enriched = f"{normalized}. {'; '.join(unique_enrichments)}"
        else:
            enriched = normalized

        return EnrichmentResult(
            original=normalized,
            enriched=enriched,
            matched_concepts=matched_concepts,
            categories=categories,
            subcategories=subcategories,
        )

    def enrich_tag(self, tag: str) -> str:
        """Обогащает одиночный тег. Возвращает enriched-строку."""
        return self.enrich_text(tag).enriched

    def enrich_interests(self, interests: dict[str, Any] | None) -> dict[str, Any]:
        """
        Обогащает структуру AIExtractedInterests / extract_interests().

        Добавляет поле ``enriched_profile_text`` — конкатенация обогащённых
        тегов для SBERT и для промпта Ollama.
        """
        if not interests:
            return {"enriched_profile_text": ""}

        parts: list[str] = []
        seen: set[str] = set()

        for field_name in ("hobbies", "skills", "topics"):
            for item in interests.get(field_name) or []:
                label = item["subcategory"] if isinstance(item, dict) else str(item)
                result = self.enrich_text(label)
                if result.enriched not in seen:
                    seen.add(result.enriched)
                    parts.append(result.enriched)

        if interests.get("occupation"):
            occ_result = self.enrich_text(str(interests["occupation"]))
            if occ_result.enriched not in seen:
                parts.append(occ_result.enriched)

        semantic_cats = interests.get("semantic_categories") or {}
        for _group, items in semantic_cats.items():
            for item in items:
                if isinstance(item, dict) and "subcategory" in item:
                    result = self.enrich_text(item["subcategory"])
                    if result.enriched not in seen:
                        parts.append(result.enriched)

        enriched_copy = dict(interests)
        enriched_copy["enriched_profile_text"] = ". ".join(parts)
        return enriched_copy

    def prepare_for_encoding(
        self, texts: list[str] | str, *, deduplicate_enrichment: bool = True,
    ) -> list[str]:
        """
        Пакетная подготовка текстов непосредственно перед SBERT.encode().

        Это главная точка перехвата в AIProfiler.calculate_text_similarity().

        Args:
            texts: Один текст или список.
            deduplicate_enrichment: Не дублировать обогащение, если оно
                совпадает с оригиналом (экономия длины контекста).

        Returns:
            list[str]: Обогащённые строки той же длины, что и вход.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not self.enabled:
            return list(texts)

        results: list[str] = []
        for text in texts:
            result = self.enrich_text(text)
            if deduplicate_enrichment and result.enriched == result.original:
                results.append(text)
            else:
                results.append(result.enriched)
        return results

    def build_profile_summary(self, interests: dict[str, Any] | None) -> str:
        """
        Краткое человекочитаемое описание интересов для LLM-промпта.
        Использует обогащённые подкатегории и направления.
        """
        if not interests:
            return "интересы не указаны"

        enriched = self.enrich_interests(interests)
        subcats: list[str] = []

        for field_name in ("hobbies", "skills", "topics"):
            for item in interests.get(field_name) or []:
                label = item["subcategory"] if isinstance(item, dict) else str(item)
                subcats.append(label)

        direction = ", ".join(subcats[:5]) if subcats else "общие интересы"
        detail = enriched.get("enriched_profile_text", "")
        if detail:
            return f"Направления: {direction}. Контекст: {detail[:400]}"
        return f"Направления: {direction}"

    @staticmethod
    def _span_overlaps(spans: list[tuple[int, int]], start: int, end: int) -> bool:
        return any(s < end and e > start for s, e in spans)

    @staticmethod
    def _is_latin_only(token: str) -> bool:
        return bool(re.fullmatch(r"[a-z]+", token, re.IGNORECASE))


@lru_cache(maxsize=1)
def get_contextual_adapter(enabled: bool = True) -> ContextualAdapter:
    """Thread-safe singleton (через lru_cache) для ContextualAdapter."""
    global _contextual_adapter_instance

    # Если адаптер еще ни разу не создавался — создаем его (это произойдет 1 раз при первом запросе)
    if _contextual_adapter_instance is None:
        # Вместо ContextualAdapter() укажи твой реальный класс, который там создается
        _contextual_adapter_instance = ContextualAdapter()

    return _contextual_adapter_instance
