#!/usr/bin/env python3
"""
Диагностика синхронизации иерархии графа интересов и резолюции тегов.
Проверяет:
1. Инициализацию иерархии в БД
2. Резолюцию тегов в существующие слаги
3. Построение весов
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from app.ai_profiler.interest_graph import (
    resolve_tag_to_slug,
    build_query_weights,
    ensure_hierarchy_seeded,
    _HIERARCHY_SEED,
)
from app.ai_profiler.semantic_ontology import SEMANTIC_ONTOLOGY
from data.session import global_init, create_session
from data.interest_hierarchy import InterestHierarchyNode
from config import config

def test_hierarchy_seed():
    """Check _HIERARCHY_SEED structure"""
    print("\n" + "="*70)
    print("TEST 1: _HIERARCHY_SEED structure")
    print("="*70)
    
    slugs_in_seed = set()
    parent_slugs_in_seed = set()
    
    for slug, name, parent_slug, weight, category in _HIERARCHY_SEED:
        slugs_in_seed.add(slug)
        if parent_slug:
            parent_slugs_in_seed.add(parent_slug)
    
    # Check if all parent references exist
    missing_parents = parent_slugs_in_seed - slugs_in_seed
    if missing_parents:
        print(f"ERROR: Missing parent slugs: {missing_parents}")
    else:
        print(f"OK: All parent references exist ({len(slugs_in_seed)} slugs)")
    
    # Check for dashes in slugs
    dashed_slugs = [s for s in slugs_in_seed if '-' in s]
    if dashed_slugs:
        print(f"WARNING: Dashed slugs (should use underscores): {dashed_slugs}")
    else:
        print("OK: No dashed slugs")

def test_semantic_ontology():
    """Check SEMANTIC_ONTOLOGY keys"""
    print("\n" + "="*70)
    print("TEST 2: SEMANTIC_ONTOLOGY keys")
    print("="*70)
    
    onto_keys = set(SEMANTIC_ONTOLOGY.keys())
    seed_slugs = {s[0] for s in _HIERARCHY_SEED}
    
    # Find keys that are in SEMANTIC_ONTOLOGY but not in HIERARCHY_SEED
    extra_keys = onto_keys - seed_slugs
    if extra_keys:
        print(f"WARNING: Keys in SEMANTIC_ONTOLOGY but not in HIERARCHY_SEED: {extra_keys}")
    else:
        print("OK: All SEMANTIC_ONTOLOGY keys exist in HIERARCHY_SEED")
    
    # Find slugs that are in HIERARCHY_SEED but not in SEMANTIC_ONTOLOGY
    missing_keys = seed_slugs - onto_keys
    if missing_keys:
        print(f"WARNING: Slugs in HIERARCHY_SEED but not in SEMANTIC_ONTOLOGY: {missing_keys}")
    else:
        print("OK: All HIERARCHY_SEED slugs exist in SEMANTIC_ONTOLOGY")

def test_db_initialization():
    """Check if database initializes correctly"""
    print("\n" + "="*70)
    print("TEST 3: Database initialization")
    print("="*70)
    
    try:
        db = create_session()
        ensure_hierarchy_seeded(db)
        
        node_count = db.query(InterestHierarchyNode).count()
        print(f"OK: Database initialized with {node_count} nodes")
        
        # Check for root nodes
        root_nodes = db.query(InterestHierarchyNode).filter(
            InterestHierarchyNode.parent_id == None
        ).all()
        print(f"  Root nodes: {len(root_nodes)} ({', '.join(n.slug for n in root_nodes[:5])}...)")
        
        db.close()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_tag_resolution():
    """Test resolve_tag_to_slug with common tags"""
    print("\n" + "="*70)
    print("TEST 4: Tag resolution accuracy")
    print("="*70)
    
    test_tags = {
        # Russian
        "разработка": "it_development",
        "программирование": "it_development",
        "python": "backend_python",
        "игры": "gaming",
        "творчество": "creativity_art",
        "искусство": "creativity_art",
        "саморазвитие": "self_development",
        "спорт": "sports_active_life",
        "музыка": "music_audio",
        "психология": "psychology_relations",
        
        # English
        "python dev": "backend_python",
        "programming": "it_development",
        "gaming": "gaming",
        "art": "creativity_art",
        
        # Mixed
        "python_dev": "backend_python",
    }
    
    correct = 0
    for tag, expected in test_tags.items():
        resolved = resolve_tag_to_slug(tag)
        status = "OK" if resolved == expected else "FAIL"
        if resolved == expected:
            correct += 1
        print(f"  [{status}] '{tag}' -> '{resolved}' (expected: '{expected}')")
    
    print(f"\nResult: {correct}/{len(test_tags)} tags resolved correctly")

def test_weight_building():
    """Test building weights from tags"""
    print("\n" + "="*70)
    print("TEST 5: Weight building")
    print("="*70)
    
    try:
        db = create_session()
        ensure_hierarchy_seeded(db)
        
        test_tags = {"python", "разработка", "flask"}
        weights = build_query_weights(db, test_tags)
        
        print(f"OK: Built weights for {len(test_tags)} tags")
        print(f"  Nodes with weights: {len(weights)}")
        print(f"  Total weight: {sum(weights.values()):.2f}")
        
        if weights:
            # Show top nodes
            sorted_nodes = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
            print("  Top 5 weighted nodes:")
            for node_id, weight in sorted_nodes:
                node = db.query(InterestHierarchyNode).filter(
                    InterestHierarchyNode.id == node_id
                ).first()
                if node:
                    print(f"    - {node.slug}: {weight:.3f}")
        
        db.close()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n" + "█"*70)
    print("NEXUS INTEREST GRAPH SYNCHRONIZATION DIAGNOSTIC")
    print("█"*70)
    
    # Initialize database first
    try:
        db_string = getattr(config, 'SQLALCHEMY_DATABASE_URI', None) or getattr(config, 'DATABASE_URL', None)
        if not db_string and hasattr(config, 'seed_db_config'):
            cfg = config.seed_db_config()
            db_string = f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
        
        if db_string:
            print(f"\nInitializing database: {db_string[:30]}...")
            global_init(db_string)
        else:
            print("\nWarning: No database URL found in config. Skipping DB tests.")
    except Exception as e:
        print(f"Warning: Could not initialize database: {e}")
    
    test_hierarchy_seed()
    test_semantic_ontology()
    test_db_initialization()
    test_tag_resolution()
    test_weight_building()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70 + "\n")
