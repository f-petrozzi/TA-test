"""
Deep diagnostic for why the transportation chunk isn't being retrieved.

This script will:
1. Find the transportation chunk in the database
2. Check if it has an embedding
3. Calculate similarity between query and chunk directly
4. Show what IS being retrieved and why
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(__file__))

from utils.rag import embed_query
from utils.supabase_client import get_supabase_client


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)


def main():
    client = get_supabase_client()

    query = "What if a student is arriving to Florida outside of the Tampa Bay Area?"

    print("=" * 80)
    print("TRANSPORTATION CHUNK RETRIEVAL DIAGNOSTIC")
    print("=" * 80)
    print(f"\nQuery: {query}\n")

    # Step 1: Find the transportation chunk
    print("STEP 1: Finding transportation chunk in database...")
    print("-" * 80)

    resp = client.table('chunks').select(
        'id, chunk_index, content, embedding, section_title, metadata, document_id'
    ).ilike('content', '%Arriving to Florida Outside of the Tampa Bay Area%').limit(3).execute()

    if not resp.data:
        print("❌ CRITICAL: Transportation chunk NOT FOUND in database!")
        print("   This means the chunk was never ingested or was deleted.")
        return

    print(f"✅ Found {len(resp.data)} matching chunk(s)")

    for chunk in resp.data:
        metadata = chunk.get('metadata') or {}
        print(f"\n📄 Chunk ID: {chunk['id']}")
        print(f"   Document ID: {chunk['document_id']}")
        print(f"   Chunk Index: {chunk['chunk_index']}")
        print(f"   Section: {chunk.get('section_title', 'N/A')}")
        print(f"   Filename: {metadata.get('filename', 'N/A')}")
        print(f"   Content preview: {chunk['content'][:150]}...")

        # Step 2: Check if chunk has embedding
        print("\nSTEP 2: Checking embedding...")
        print("-" * 80)

        embedding = chunk.get('embedding')
        if not embedding:
            print("❌ CRITICAL: Chunk has NO EMBEDDING!")
            print("   This chunk cannot be retrieved via vector search.")
            print("   You need to re-ingest this document.")
            continue

        # Handle different embedding formats from database
        if isinstance(embedding, str):
            print(f"⚠️  Embedding is stored as STRING (length: {len(embedding)})")
            print("   Attempting to parse as JSON...")
            import json
            try:
                embedding = json.loads(embedding)
            except:
                print("   ❌ Failed to parse embedding string")
                continue

        # Flatten if nested
        if embedding and isinstance(embedding[0], (list, tuple)):
            embedding = embedding[0]

        # Debug: Show raw embedding info before conversion
        print(f"   Raw embedding type: {type(embedding)}")
        print(f"   Raw embedding length: {len(embedding)}")
        if embedding:
            print(f"   First element type: {type(embedding[0])}")
            print(f"   First 3 elements: {embedding[:3]}")

        # Convert to floats
        try:
            embedding = [float(x) for x in embedding]
        except (TypeError, ValueError) as e:
            print(f"❌ CRITICAL: Cannot convert embedding to floats: {e}")
            print(f"   Embedding type: {type(embedding)}")
            if embedding:
                print(f"   First element type: {type(embedding[0])}")
                print(f"   First element: {embedding[0]}")
            continue

        print(f"✅ Chunk has embedding (dimension: {len(embedding)})")
        print(f"   First 3 values after conversion: {embedding[:3]}")

        # Step 3: Calculate direct similarity
        print("\nSTEP 3: Testing direct similarity with query...")
        print("-" * 80)

        # Test both formats
        query_clean = query
        query_with_prefix = f"title: none | text: {query}"

        emb_clean = list(embed_query(query_clean))
        emb_prefix = list(embed_query(query_with_prefix))

        # Debug query embeddings
        print(f"   Query embedding (clean) type: {type(emb_clean)}")
        print(f"   Query embedding (clean) dimension: {len(emb_clean)}")
        print(f"   Query embedding (clean) first 3: {emb_clean[:3]}")
        print(f"   Chunk embedding dimension: {len(embedding)}")
        print(f"   Chunk embedding first 3: {embedding[:3]}")

        sim_clean = cosine_similarity(emb_clean, embedding)
        sim_prefix = cosine_similarity(emb_prefix, embedding)

        print(f"Similarity (clean query):        {sim_clean:.6f}")
        print(f"Similarity (with title prefix):  {sim_prefix:.6f}")
        print(f"Difference:                      {abs(sim_clean - sim_prefix):.6f}")

        if sim_clean < 0.3 and sim_prefix < 0.3:
            print("\n⚠️  WARNING: Both similarities are LOW (< 0.3)")
            print("   This suggests the chunk embedding doesn't match the query well.")
            print("   Possible causes:")
            print("   - Chunk was embedded with different format than current queries")
            print("   - Embedding model mismatch")
            print("   - Chunk content has too much noise/boilerplate")
        elif sim_clean > 0.5 or sim_prefix > 0.5:
            print("\n✅ Good similarity (> 0.5) - chunk should rank well")
        else:
            print("\n⚠️  Moderate similarity (0.3-0.5) - might get buried by other results")

    # Step 4: Show what IS being retrieved
    print("\n\nSTEP 4: What IS being retrieved instead?")
    print("=" * 80)

    # Use current format (with prefix)
    query_embedding = list(embed_query(f"title: none | text: {query}"))

    payload = {
        "query_embedding": query_embedding,
        "match_count": 10,
    }

    resp = client.rpc("match_document_chunks", payload).execute()
    hits = getattr(resp, "data", []) or []

    print(f"\nTop 10 retrieved chunks:")
    print("-" * 80)

    for i, hit in enumerate(hits[:10], 1):
        meta = hit.get('metadata') or {}
        title = meta.get('section_title') or meta.get('filename') or 'Unknown'
        similarity = hit.get('similarity', 0.0)
        content_preview = (hit.get('content') or '')[:100]

        # Check if this is the transportation chunk
        is_transport = 'arriving to florida outside' in content_preview.lower()
        marker = "🎯 THIS IS IT!" if is_transport else ""

        print(f"\n{i}. {title} {marker}")
        print(f"   Similarity: {similarity:.6f}")
        print(f"   Preview: {content_preview}...")

        if is_transport:
            print(f"   ✅ Found at rank #{i}")

    # Check if transportation chunk is in top 10
    transport_in_top10 = any('arriving to florida outside' in (hit.get('content') or '').lower() for hit in hits[:10])

    if not transport_in_top10:
        print("\n\n❌ CRITICAL: Transportation chunk NOT in top 10 results!")
        print("   This explains why it's not being retrieved.")
        print("\n   Possible fixes:")
        print("   1. Check chunk embedding format matches query format")
        print("   2. Increase SUPABASE_INITIAL_MATCH_COUNT to retrieve more candidates")
        print("   3. Re-ingest with consistent embedding format")
        print("   4. Examine chunk content for issues (too much boilerplate, etc.)")
    else:
        print("\n\n🤔 Transportation chunk IS in top 10, but might be getting filtered during reranking")
        print("   Check reranking parameters and cross-encoder behavior")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
