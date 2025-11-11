"""
Test script to compare retrieval quality with and without title prefixes.

Run this to see empirically which format works better for your use case.

Usage:
    python test_embedding_formats.py
"""

import os
import sys
from typing import List, Dict, Any

# Set environment before imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, os.path.dirname(__file__))

from utils.rag import embed_query
from utils.supabase_client import get_supabase_client

# Test queries that should retrieve specific content
TEST_QUERIES = [
    {
        "query": "What if a student is arriving to Florida outside of the Tampa Bay Area?",
        "expected_keywords": ["redcoach", "flixbus", "greyhound", "bus"],
        "expected_title": "Transportation"
    },
    {
        "query": "What are the orientation dates for international students?",
        "expected_keywords": ["orientation", "glo-bull", "international"],
        "expected_title": "Orientation"
    },
    {
        "query": "How do I request an official transcript?",
        "expected_keywords": ["transcript", "registrar", "request"],
        "expected_title": "Transcript"
    },
]


def retrieve_with_format(query: str, use_title_prefix: bool) -> List[Dict[str, Any]]:
    """Retrieve chunks using specified embedding format."""
    client = get_supabase_client()

    # Format query based on test mode
    if use_title_prefix:
        formatted_query = f"title: none | text: {query}"
    else:
        formatted_query = query

    # Get embedding
    embedding = list(embed_query(formatted_query))

    # Retrieve top 10 matches
    payload = {
        "query_embedding": embedding,
        "match_count": 10,
    }

    resp = client.rpc("match_document_chunks", payload).execute()
    return getattr(resp, "data", []) or []


def evaluate_results(hits: List[Dict[str, Any]], test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate retrieval quality for a test case."""
    if not hits:
        return {
            "found_expected": False,
            "rank": None,
            "top_score": 0.0,
            "has_keywords": False
        }

    expected_keywords = test_case["expected_keywords"]
    expected_title = test_case["expected_title"].lower()

    # Check if expected content is in top results
    found_rank = None
    has_keywords = False

    for i, hit in enumerate(hits[:5], 1):
        content = (hit.get("content") or "").lower()
        meta = hit.get("metadata") or {}
        title = (meta.get("section_title") or meta.get("filename") or "").lower()

        # Check if this chunk contains expected keywords
        if any(kw in content for kw in expected_keywords):
            has_keywords = True
            if found_rank is None:
                found_rank = i

        # Check if title matches
        if expected_title in title and found_rank is None:
            found_rank = i

    top_score = hits[0].get("similarity", 0.0) if hits else 0.0

    return {
        "found_expected": found_rank is not None,
        "rank": found_rank,
        "top_score": float(top_score),
        "has_keywords": has_keywords,
        "top_result_title": (hits[0].get("metadata") or {}).get("section_title", "Unknown") if hits else "None"
    }


def run_comparison():
    """Run comparison test between title prefix and clean embeddings."""
    print("=" * 80)
    print("EMBEDDING FORMAT COMPARISON TEST")
    print("=" * 80)
    print("\nTesting two embedding formats:")
    print("  1. WITH title prefix:    'title: none | text: {query}'")
    print("  2. WITHOUT title prefix:  '{query}' (clean)")
    print("\n" + "=" * 80)

    results_with_title = []
    results_without_title = []

    for i, test_case in enumerate(TEST_QUERIES, 1):
        query = test_case["query"]
        print(f"\n\n📝 TEST {i}/{len(TEST_QUERIES)}")
        print(f"Query: {query}")
        print(f"Expected: {test_case['expected_title']} document with keywords: {', '.join(test_case['expected_keywords'])}")
        print("-" * 80)

        # Test WITH title prefix
        print("\n🔹 WITH title prefix:")
        hits_with = retrieve_with_format(query, use_title_prefix=True)
        eval_with = evaluate_results(hits_with, test_case)
        results_with_title.append(eval_with)

        print(f"   Found expected content: {'✅ YES' if eval_with['found_expected'] else '❌ NO'}")
        if eval_with['rank']:
            print(f"   Rank: #{eval_with['rank']}")
        print(f"   Top result: {eval_with['top_result_title']}")
        print(f"   Top score: {eval_with['top_score']:.4f}")

        # Test WITHOUT title prefix
        print("\n🔹 WITHOUT title prefix (clean):")
        hits_without = retrieve_with_format(query, use_title_prefix=False)
        eval_without = evaluate_results(hits_without, test_case)
        results_without_title.append(eval_without)

        print(f"   Found expected content: {'✅ YES' if eval_without['found_expected'] else '❌ NO'}")
        if eval_without['rank']:
            print(f"   Rank: #{eval_without['rank']}")
        print(f"   Top result: {eval_without['top_result_title']}")
        print(f"   Top score: {eval_without['top_score']:.4f}")

        # Compare
        if eval_with['found_expected'] and eval_without['found_expected']:
            if eval_without['rank'] < eval_with['rank']:
                print("\n   💡 Winner: CLEAN (better rank)")
            elif eval_with['rank'] < eval_without['rank']:
                print("\n   💡 Winner: WITH PREFIX (better rank)")
            else:
                print("\n   💡 Tie: Both found at same rank")
        elif eval_without['found_expected'] and not eval_with['found_expected']:
            print("\n   💡 Winner: CLEAN (only this format found it)")
        elif eval_with['found_expected'] and not eval_without['found_expected']:
            print("\n   💡 Winner: WITH PREFIX (only this format found it)")
        else:
            print("\n   💡 Both failed to find expected content")

    # Summary
    print("\n\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    with_success = sum(1 for r in results_with_title if r['found_expected'])
    without_success = sum(1 for r in results_without_title if r['found_expected'])

    with_avg_rank = sum(r['rank'] for r in results_with_title if r['rank']) / max(with_success, 1)
    without_avg_rank = sum(r['rank'] for r in results_without_title if r['rank']) / max(without_success, 1)

    print(f"\n🔹 WITH title prefix:")
    print(f"   Success rate: {with_success}/{len(TEST_QUERIES)} ({with_success/len(TEST_QUERIES)*100:.0f}%)")
    print(f"   Avg rank (when found): {with_avg_rank:.2f}")

    print(f"\n🔹 WITHOUT title prefix (clean):")
    print(f"   Success rate: {without_success}/{len(TEST_QUERIES)} ({without_success/len(TEST_QUERIES)*100:.0f}%)")
    print(f"   Avg rank (when found): {without_avg_rank:.2f}")

    print("\n" + "=" * 80)
    print("🎯 RECOMMENDATION:")
    if without_success > with_success or (without_success == with_success and without_avg_rank < with_avg_rank):
        print("   ✅ Use CLEAN embeddings (no title prefix)")
        print("   Reason: Better retrieval quality and/or ranking")
    elif with_success > without_success:
        print("   ✅ Keep title prefix")
        print("   Reason: Better retrieval quality")
    else:
        print("   ⚖️  Both perform equally - slight preference for CLEAN (simpler)")
    print("=" * 80)


if __name__ == "__main__":
    try:
        run_comparison()
    except Exception as e:
        print(f"\n❌ Error running comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
