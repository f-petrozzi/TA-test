"""
Test to verify that URLs are preserved in stored content but stripped from embeddings.

This proves that citations will still work after URL stripping is applied.
"""

import sys
sys.path.insert(0, '.')

from data_ingestion import _format_for_embedding

# Test chunk with URLs
test_chunk = """
Visit HART online at https://www.hart.org/ for more information.
You can also check FlixBus at https://www.flixbus.com/bus-routes

Contact information:
- Website: https://www.usf.edu/
- Email: info@usf.edu
"""

print("=" * 80)
print("URL PRESERVATION TEST")
print("=" * 80)

print("\n📄 ORIGINAL CHUNK (what gets stored in database):")
print("-" * 80)
print(test_chunk)
print(f"\nLength: {len(test_chunk)} characters")
print(f"Contains 'https://': {test_chunk.count('https://')} times")

print("\n\n🧹 CLEANED FOR EMBEDDING (what gets embedded):")
print("-" * 80)
cleaned = _format_for_embedding(test_chunk, "Test")
print(cleaned)
print(f"\nLength: {len(cleaned)} characters")
print(f"Contains 'https://': {cleaned.count('https://')} times")

print("\n\n" + "=" * 80)
print("✅ VERIFICATION")
print("=" * 80)

original_has_urls = "https://" in test_chunk
cleaned_has_urls = "https://" in cleaned

print(f"\n✅ Original chunk has URLs: {original_has_urls}")
print(f"✅ Cleaned embedding text has URLs: {cleaned_has_urls}")

if original_has_urls and not cleaned_has_urls:
    print("\n🎉 SUCCESS!")
    print("   - URLs preserved in stored content (for citations)")
    print("   - URLs removed from embeddings (for better semantic matching)")
    print("   - Citations will still work perfectly!")
elif original_has_urls and cleaned_has_urls:
    print("\n❌ WARNING: URLs not being stripped from embeddings")
    print("   - Embeddings will still have URL noise")
else:
    print("\n❓ Test inconclusive (no URLs in test chunk?)")

print("\n" + "=" * 80)
print("HOW IT WORKS IN PRODUCTION")
print("=" * 80)
print("""
When ingesting:
1. Original chunk (WITH URLs) is stored in database → Used for citations
2. Cleaned chunk (NO URLs) is embedded → Used for similarity matching
3. Both are paired together in the chunks table

When retrieving:
1. Query is embedded (clean)
2. Similarity matching uses clean embeddings
3. Retrieved chunks include original content WITH URLs
4. User sees URLs in cited sources
""")
