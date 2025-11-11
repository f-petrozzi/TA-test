# Re-Ingestion Instructions

## Why Re-Ingest?

The title prefix has been removed from both ingestion and query code. To apply these changes, you need to re-ingest your corpus so all embeddings are regenerated with the new clean format.

## Expected Impact

After re-ingestion, similarity scores should improve:
- **Before**: Transportation chunks scored 0.38-0.41 (ranked 11+)
- **Expected After**: 0.50+ similarity (ranked in top 10)

## Steps to Re-Ingest

### 1. Backup Current Database (Optional but Recommended)

```bash
# This is optional - you can skip if you're confident
# Use Supabase dashboard to export data, or:
# pg_dump -h <supabase-host> -U postgres -d postgres > backup.sql
```

### 2. Run the Ingestion Script

```bash
python data_ingestion.py --source data/raw
```

**Options:**
- `--dry-run`: Preview what will be ingested without actually doing it
- `--skip-unchanged`: Skip documents that haven't changed (based on checksum)
- `--chunk 700`: Chunk size in characters (default: 700)
- `--overlap 220`: Overlap between chunks (default: 220)

**Example with dry run first:**
```bash
# Preview
python data_ingestion.py --source data/raw --dry-run

# Actually ingest
python data_ingestion.py --source data/raw
```

### 3. Verify Re-Ingestion

Run the diagnostic script to verify improved similarity scores:

```bash
python diagnose_transportation_retrieval.py
```

**Expected output:**
- Chunk similarity should improve from 0.38-0.41 to 0.50+
- Transportation chunks should appear in top 10 results
- "What IS being retrieved" section should show transportation chunks

### 4. Test in Streamlit

```bash
streamlit run app.py
```

Ask: "What if a student is arriving to Florida outside of the Tampa Bay Area?"

**Expected behavior:**
- Should retrieve Transportation document
- Should mention RedCoach, FlixBus, Greyhound
- Should cite correct source (not "Unofficial Transcripts"!)

## Troubleshooting

### If ingestion fails:

1. **Check environment variables** are set:
   ```bash
   # In .env or environment:
   SUPABASE_URL=...
   SUPABASE_SERVICE_KEY=...
   HUGGINGFACEHUB_API_TOKEN=...
   ```

2. **Check data source** exists:
   ```bash
   ls data/raw/
   ```

3. **Check Supabase connection**:
   ```python
   from utils.supabase_client import get_supabase_client
   client = get_supabase_client()
   print(client.table('documents').select('id').limit(1).execute())
   ```

### If similarity scores don't improve:

This could indicate:
1. Chunks have too much noise (URLs, boilerplate)
2. Need better chunking strategy
3. Different embedding model might help
4. Content quality issues

## Monitoring Progress

The ingestion script will output:
- `[ingest] {filename}` - Successfully ingested
- `[skip unchanged] {filename}` - Skipped (already up-to-date)
- `[skip empty] {filename}` - Skipped (empty file)
- Progress indicators for embedding batches

## Estimated Time

- **Small corpus (< 100 docs)**: ~2-5 minutes
- **Medium corpus (100-500 docs)**: ~10-20 minutes
- **Large corpus (500+ docs)**: ~30-60 minutes

Time depends on:
- Number of documents
- Document sizes
- HuggingFace API response time
- Network speed

## What Changed?

**Before (Old Format):**
```python
# Chunks embedded as:
"title: New Students | text: {content}"

# Queries embedded as:
"title: none | text: {query}"
```

**After (New Format):**
```python
# Both chunks and queries embedded as:
"{text}"  # Clean, no prefix
```

This ensures perfect semantic alignment between chunks and queries, improving retrieval accuracy.

---

**Questions?** See the commit message in `7869e55` for full technical details.
