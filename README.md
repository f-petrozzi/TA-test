## USF Onboarding Assistant — Cloud Deployment Checklist

### 1. Architecture (Production-ready for the final project)
- **Frontend**: Streamlit app (`app.py`) served on Streamlit Community Cloud.
- **LLM**: Azure OpenAI (Phi-4) via `AZURE_OPENAI_DEPLOYMENT`.
- **Embeddings**: Hugging Face Inference API (`google/embeddinggemma-300m`).
- **Vector Store**: Supabase Postgres + pgvector (`chunks` table + `match_document_chunks()` RPC).
- **Relational Data**: Supabase tables for users, chat_sessions, messages, audit_logs.
- **Ingestion**: `data_ingestion.py` cleans Markdown, builds embeddings with Hugging Face, writes to Supabase.
- **Agent Tooling**: `utils/mcp.SimpleMCPClient` exposes MCP-style `retrieve_context`/`log_interaction` tools plus Gmail & Google Calendar helpers wired into the UI.

### 2. Environment Variables
Copy `.env.example` to `.env` and populate the values below (also mirror them in Streamlit secrets):

```
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=phi-4

HUGGINGFACEHUB_API_TOKEN=
HUGGINGFACE_EMBEDDING_MODEL=google/embeddinggemma-300m

SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_ANON_KEY=                   # optional, for client-side reads
SUPABASE_DOCUMENTS_TABLE=documents
SUPABASE_CHUNKS_TABLE=chunks
SUPABASE_SESSIONS_TABLE=chat_sessions
SUPABASE_MESSAGES_TABLE=messages
SUPABASE_USERS_TABLE=users
SUPABASE_AUDIT_TABLE=audit_logs
SUPABASE_MATCH_FUNCTION=match_document_chunks

SESSION_TOKEN_LIMIT=1500

# Google Workspace (OAuth refresh-token flow)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REFRESH_TOKEN=
GOOGLE_TOKEN_URI=https://oauth2.googleapis.com/token  # optional override
```

### 3. Supabase Schema
Run once (SQL editor):

```sql
create extension if not exists vector;

create table if not exists users (
  id uuid primary key,
  username text unique not null,
  email text,
  salt text not null,
  pwd_hash text not null,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists documents (
  id uuid primary key,
  title text,
  source_path text,
  category text,
  checksum text unique,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists chunks (
  id uuid primary key,
  document_id uuid references documents(id) on delete cascade,
  chunk_index int,
  content text,
  section_title text,
  metadata jsonb,
  embedding vector(3072),
  chunk_fp text unique,
  created_at timestamptz default now()
);

create table if not exists chat_sessions (
  id uuid primary key,
  user_id uuid references users(id) on delete cascade,
  session_name text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists messages (
  id uuid primary key,
  session_id uuid references chat_sessions(id) on delete cascade,
  role text,
  content text,
  tokens_in int,
  tokens_out int,
  created_at timestamptz default now()
);

create table if not exists audit_logs (
  id uuid primary key,
  session_id uuid references chat_sessions(id) on delete cascade,
  event_type text,
  payload jsonb,
  created_at timestamptz default now()
);
```

Vector similarity RPC:

```sql
create or replace function match_document_chunks(
  query_embedding vector(3072),
  match_count int default 6,
  filter jsonb default '{}'::jsonb
)
returns table (
  id uuid,
  content text,
  section_title text,
  metadata jsonb,
  similarity double precision
) language plpgsql as $$
begin
  return query
  select
    c.id,
    c.content,
    c.section_title,
    c.metadata,
    1 - (c.embedding <=> query_embedding) as similarity
  from chunks c
  where (filter = '{}'::jsonb) or (c.metadata @> filter)
  order by c.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

### 4. Data Ingestion Workflow
1. Export USF content to Markdown (`data/raw`).
2. Ensure Azure Phi-4 deployment and Hugging Face token exist; update `.env`.
3. Run:
   ```
   python data_ingestion.py \
     --source data/raw \
     --skip-unchanged        # optional
   ```
   - Cleans + chunks Markdown.
   - Calls Hugging Face embeddings (batch).
   - Upserts documents + chunks into Supabase (pgvector).

### 5. Assisted Actions (MCP-style)
- Click **📧 Email Assistant** below the chat to open a guided form. Enter the student’s email, subject, and their inquiry, then press **Generate Draft** to have the bot craft a reply from the USF corpus. From there you can:
  - Edit the text manually and save it.
  - Provide optional AI edit instructions and press **Apply AI Edit** to regenerate.
  - Press **Send Email** to have the bot deliver the message through Gmail immediately.
- Click **📅 Meeting Assistant** to enter meeting summary, ISO start time (e.g., `2024-09-12T14:00-04:00`), duration, attendees, and notes. Press **Check Availability** to query Google Calendar; the bot reports whether the slot is open (and suggests the next free window if not). When you’re happy, press **Create Event** to insert it into Google Calendar.
- Both assistants keep the latest draft/plan visible in the sidebar for context, while confirmations and results remain in the chat transcript—no manual copying or sidebar clicks required.

### 6. Running the App Locally
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Make sure `.env` is present in the project root before launching; `python-dotenv` will load it automatically.

### 7. Deployment Notes
- Store **service role key** as Streamlit secret; never expose to users.
- On Streamlit Cloud, add `requirements.txt` and `.streamlit/secrets.toml` with the env keys.
- Ingestion should run locally or via GitHub Action; the deployed Streamlit app only reads vectors and writes chat state.
- Google OAuth refresh tokens should be generated once using the shared admin account (Scopes listed in `utils/google_tools.py`). Add `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN`, and (optionally) `GOOGLE_TOKEN_URI` to `.env` and Streamlit secrets.

### 8. MCP and Agent Hooks
- Retrieval metadata (`hits`) returned from `generate_with_rag` is logged to `audit_logs`. An MCP server can expose tools:
  1. `retrieve_context` → call Supabase RPC.
  2. `log_interaction` → write to `audit_logs`.
  3. Gmail / Google Calendar tools (`list_calendar_events`, `list_recent_emails`, `send_email`, `create_event`).
- Streamlit already consumes these via `SimpleMCPClient`, so hooking MCP later simply means swapping the client implementation.

### 9. Testing Checklist
- [ ] `python data_ingestion.py --dry-run` succeeds (chunks counted).
- [ ] Full ingestion populates Supabase tables.
- [ ] Streamlit login/register persists users in Supabase.
- [ ] Sessions/messages survive refresh & appear in Supabase.
- [ ] Chat answers include sources from Supabase chunks.
- [ ] Audit logs record prompt/response metadata.

### 10. Troubleshooting
- **Missing env keys** → Streamlit will surface runtime error (see sidebar).
- **Vector RPC not found** → ensure `match_document_chunks` deployed with correct signature.
- **Hugging Face throttling** → lower batch size or add retries/backoff on the inference API.
- **pgvector dimension mismatch** → match table dimension to embedding model (3072 for `google/embeddinggemma-300m`).

### 11. MCP Integration (Beginner Friendly)
- `utils/mcp.py` contains `SimpleMCPClient`, a tiny stand-in for a Model Context Protocol client. It exposes the two “tools” our app needs today:
  1. `retrieve_context(query, match_count)` → wraps the existing Supabase/pgvector search helper.
  2. `log_interaction(session_id, event_type, payload)` → writes to the `audit_logs` table.
  3. `list_calendar_events` / `list_recent_emails` / `send_email` / `create_event` → Google Calendar + Gmail via OAuth refresh tokens so you can demo MCP-style tool use with real accounts.
- `app.py` instantiates `SimpleMCPClient` once and passes it to `generate_with_rag`, so swapping in a real MCP server later only requires changing the client implementation—not the UI logic.
- To connect to a future MCP service, replace `SimpleMCPClient` with a class that calls your MCP transport layer but keeps the same two method signatures.

This README covers all changes needed to meet the final project requirements (cloud LLM, managed vector DB, production-minded persistence, audit logging). Commit this doc alongside code changes so teammates and graders can follow the deployment plan.
