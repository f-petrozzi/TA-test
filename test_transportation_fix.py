"""
Test the improved transportation query retrieval.

This script tests if the fixes (increased initial count + query augmentation)
successfully retrieve the transportation chunk.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from utils.rag import retrieve_matches, _augment_query

query = "What if a student is arriving to Florida outside of the Tampa Bay Area?"

print("=" * 80)
print("TESTING IMPROVED TRANSPORTATION RETRIEVAL")
print("=" * 80)

# Show augmented query
augmented = _augment_query(query)
print(f"\nOriginal Query:\n{query}")
print(f"\nAugmented Query:\n{augmented}")
print("\n" + "=" * 80)

# Retrieve with new settings
print("\nRetrieving chunks with improved settings...")
print("  - Initial retrieval count: 100 (was 50)")
print("  - Query augmentation: enabled")
print("-" * 80)

hits = retrieve_matches(
    augmented,
    match_count=10,
    embedding_text=query,  # Embed original query, use augmented for retrieval
)

print(f"\n✅ Retrieved {len(hits)} chunks\n")

# Check if transportation chunk is in results
found_transport = False
transport_rank = None

for i, hit in enumerate(hits, 1):
    meta = hit.get('meta', {})
    title = meta.get('section_title') or meta.get('filename', 'Unknown')
    doc = hit.get('doc', '')
    rerank_score = hit.get('rerank_score', hit.get('score', 0))

    # Check if this is the transportation chunk
    is_transport = ('redcoach' in doc.lower() or
                   'flixbus' in doc.lower() or
                   'arriving to florida outside' in doc.lower())

    marker = "🎯" if is_transport else "  "

    print(f"{marker} {i}. {title}")
    print(f"   Rerank Score: {rerank_score:.4f}")
    print(f"   Preview: {doc[:100]}...")

    if is_transport and not found_transport:
        found_transport = True
        transport_rank = i

    print()

print("=" * 80)
if found_transport:
    print(f"✅ SUCCESS: Transportation chunk found at rank #{transport_rank}!")
    print(f"   The fixes improved retrieval - chunk is now in top 10.")
else:
    print(f"❌ FAILED: Transportation chunk still not in top 10")
    print(f"   Additional fixes may be needed:")
    print(f"   - Strip URLs from chunks during ingestion")
    print(f"   - Improve chunking to keep Q&A together")
    print(f"   - Further increase initial retrieval count")
print("=" * 80)
