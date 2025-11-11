# Universal Retrieval Improvements

This document outlines universal improvements to the RAG system that benefit **all queries**, not specific edge cases.

---

## 🎯 Philosophy: Universal > Specific

**Bad Approach:** Query-specific augmentation (e.g., "if query mentions 'bus' add 'RedCoach, FlixBus'")
- Brittle and doesn't scale
- Requires maintaining rules for every topic
- Masks underlying problems

**Good Approach:** Universal improvements to content quality and retrieval pipeline
- Benefits all queries equally
- Scales naturally
- Fixes root causes, not symptoms

---

## ✅ Universal Improvements Applied

### **1. Increased Initial Retrieval Count: 50 → 100**

**File:** `utils/rag.py` (line 36)

**Problem:**
- Borderline-relevant chunks (similarity 0.38-0.45) filtered out before reranking
- Reranker never got a chance to properly score them

**Solution:**
- Retrieve 100 candidates instead of 50
- Let cross-encoder reranker make the final decision
- Wider funnel = better recall

**Impact:**
- ✅ Universal: Helps all queries that have relevant chunks ranked 50-100
- ✅ No downsides: Minimal performance impact (reranking is on 20 candidates, not 100)
- ✅ No re-ingestion required

**Expected Improvement:**
- Chunks ranked 50-100 now have a chance to be reranked and promoted
- Recall improves without sacrificing precision

---

### **2. Strip URLs from Embeddings**

**File:** `data_ingestion.py` (_format_for_embedding function)

**Problem:**
- Analysis showed 35-62% URL content in chunks
- URLs like `https://www.example.com/very/long/path` add noise, no semantic value
- Dilutes embedding quality for ALL chunks

**Solution:**
```python
# Before embedding, strip URLs but keep semantic text
cleaned = re.sub(r'https?://\S+', '', text)
# "Visit HART online at https://hart.org" → "Visit HART online at"
```

**Key Points:**
- URLs stripped ONLY from embedding text
- Original content with URLs preserved in database for citations
- Markdown link cleanup to avoid orphaned brackets

**Impact:**
- ✅ Universal: Reduces noise for all chunks across entire corpus
- ✅ Should increase similarity scores universally
- ✅ No loss of information (URLs still available for citations)
- ⚠️  Requires re-ingestion to apply

**Expected Improvement:**
- Embeddings focus on semantic content, not URL noise
- Similarity scores should increase 0.05-0.15 across the board
- Better discrimination between truly relevant and irrelevant chunks

---

## 📊 Expected Impact on Transportation Query

Using the transportation query as a test case:

| Metric | Before | After Universal Fixes |
|--------|--------|----------------------|
| **Chunk 12 Similarity** | 0.408 | 0.45-0.50 (URL noise removed) |
| **Initial Retrieval Rank** | ~15-20 | Same (depends on similarity) |
| **Makes it to Reranking?** | ❌ No (filtered at 50) | ✅ Yes (included in 100) |
| **Final Rank After Reranking** | N/A | 3-7 (cross-encoder should recognize relevance) |

---

## 🔄 Next Steps

### 1. Re-Ingest Corpus with URL Stripping

```bash
python data_ingestion.py --source data/raw
```

This will:
- Generate cleaner embeddings without URL noise
- Increase similarity scores universally
- Take 5-20 minutes depending on corpus size

### 2. Test Multiple Queries (Not Just Transportation)

Test diverse queries to verify universal improvement:

```python
test_queries = [
    "What if a student is arriving to Florida outside of the Tampa Bay Area?",  # Transportation
    "What are the orientation dates for international students?",  # Orientation
    "How do I request an official transcript?",  # Registrar
    "What are the parking options at USF Tampa?",  # Campus services
    "When is the add/drop deadline?",  # Academic calendar
]
```

Expected: All queries should show improved retrieval quality.

### 3. Run Diagnostic

```bash
python diagnose_transportation_retrieval.py
```

Expected output:
- Chunk 12 similarity: **0.45-0.50** (up from 0.408)
- Chunk appears in "Top 10 retrieved chunks"
- Rerank score competitive with other chunks

---

## 🚫 What We Avoided

### Query-Specific Augmentation (Removed)

We initially tried this:
```python
# DON'T DO THIS - too specific
if "arriving" in query and "florida" in query:
    query += "bus RedCoach FlixBus Greyhound"
```

**Why it's bad:**
- Only fixes one specific query type
- Requires rules for every topic (orientation, transcripts, housing, etc.)
- Becomes unmaintainable with hundreds of topics
- Masks real problems (noisy embeddings, poor chunking)

### The Better Alternative

Fix the root causes:
- ✅ Noisy embeddings → Strip URLs
- ✅ Early filtering → Increase initial retrieval count
- ✅ Poor chunks → Better chunking (future improvement)

These fixes help **ALL queries**, not just one.

---

## 🔮 Future Universal Improvements

These would require more significant changes:

### 1. Semantic Chunking
**Current:** Fixed-size chunks (700 chars) split arbitrarily
**Better:** Semantic boundaries (preserve Q&A pairs, section integrity)
**Impact:** Universal improvement to chunk quality

### 2. Hybrid Search
**Current:** Pure vector search
**Better:** Vector search + BM25 keyword search
**Impact:** Better handling of specific terms (names, dates, codes)

### 3. Better Cross-Encoder
**Current:** `ms-marco-MiniLM-L-6-v2`
**Better:** Domain-specific reranker or larger model
**Impact:** More accurate final ranking

### 4. Chunk Deduplication
**Current:** Fingerprint-based dedup during ingestion
**Better:** Semantic dedup (remove near-duplicate chunks)
**Impact:** Less redundancy, higher quality results

---

## 📝 Summary

**Universal improvements applied:**
1. ✅ Increased initial retrieval count (50 → 100)
2. ✅ Strip URLs from embeddings

**Expected results:**
- Better recall (more relevant chunks reach reranking)
- Cleaner embeddings (less URL noise)
- Improved similarity scores universally
- Better retrieval quality for ALL queries

**Next step:**
- Re-ingest corpus with URL stripping
- Test across multiple query types
- Verify universal improvement

**Philosophy:**
- Fix root causes, not symptoms
- Universal solutions > specific patches
- Scale naturally with corpus growth
