"""
Debug script to examine the actual content of transportation chunks
to identify if there's noise diluting the semantic signal.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from utils.supabase_client import get_supabase_client

client = get_supabase_client()

# Get the transportation chunks
resp = client.table('chunks').select(
    'chunk_index, content, section_title'
).ilike('content', '%Arriving to Florida Outside%').execute()

print("=" * 80)
print("TRANSPORTATION CHUNK CONTENT ANALYSIS")
print("=" * 80)

for row in resp.data:
    print(f"\n{'='*80}")
    print(f"Chunk {row['chunk_index']}: {row.get('section_title', 'N/A')}")
    print(f"{'='*80}")
    print(f"Length: {len(row['content'])} characters")
    print(f"\nFull Content:")
    print("-" * 80)
    print(row['content'])
    print("-" * 80)

    # Analyze content composition
    lines = row['content'].split('\n')
    url_lines = [l for l in lines if 'http' in l or 'www.' in l]
    header_lines = [l for l in lines if l.strip().startswith('#')]
    list_lines = [l for l in lines if l.strip().startswith('*') or l.strip().startswith('-')]

    print(f"\nContent Composition:")
    print(f"  Total lines: {len(lines)}")
    print(f"  Lines with URLs: {len(url_lines)}")
    print(f"  Header lines: {len(header_lines)}")
    print(f"  List items: {len(list_lines)}")
    print(f"  URL ratio: {len(url_lines)/len(lines)*100:.1f}%")

    # Check for the key question
    if "Arriving to Florida Outside" in row['content']:
        print(f"\n✅ Contains 'Arriving to Florida Outside' text")
        # Find the sentence
        for line in lines:
            if "Arriving to Florida Outside" in line:
                print(f"  Line: {line}")

    # Check for bus company names
    bus_companies = ['RedCoach', 'FlixBus', 'Greyhound', 'bus']
    found = [b for b in bus_companies if b.lower() in row['content'].lower()]
    print(f"\n{'✅' if found else '❌'} Bus companies mentioned: {', '.join(found) if found else 'NONE'}")
