# Go-Care AI Ticket Recommendation System

Helps support agents solve a new ticket faster: it finds the most similar
**resolved** historical tickets and returns the **resolution articles** that
were written on them, in full.

```
new ticket ─► clean text ─► bge-m3 embedding ─► metadata filter ─► exact cosine ranking
          ─► top 5 resolved tickets ─► all their articles (looked up by TicketNumber) ─► JSON / Streamlit UI
          ─► (optional) AI advisor: all 5 tickets + all their articles ─► LLM ─► grounded next step
```

Retrieval never generates or invents anything: every result is a stored
ticket or article with the similarity score that led to it. The optional AI
advisor must cite those tickets for every step it proposes. Steps without a
valid source are discarded.

---

## 1. Quick start

```bash
git clone <repo-url> go-care-ai && cd go-care-ai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch (no CUDA download)
pip install -r requirements.txt

# Put the data in "Raw Data/" (it is git-ignored, so copy it with scp/rsync):
#   Raw Data/Tickets_oct26.csv
#   Raw Data/Article_*.csv        (optional, can be added later)

python build_index.py            # ONE command: clean -> embed -> index -> articles
streamlit run app.py --server.address 0.0.0.0 --server.port 8501    # the agent UI
```

### Running on the CPU server over SSH

The server has no GPU (64 GB RAM). `config.yaml` is set for that:
`embedding.device: cpu`, `batch_size: 16`, `cpu_threads: 0` (all cores).
bge-m3 is a large model, so the **first build of ~909k tickets takes many
hours on CPU**. The log prints the throughput and an ETA after every shard of
20,000 tickets. Memory is not a concern: the model takes ~2–3 GB, the exact
FAISS index ~3.7 GB and the float16 embeddings ~1.9 GB.

Run it detached so a dropped SSH session does not stop it:

```bash
nohup python build_index.py > artifacts/build.out 2>&1 &
tail -f artifacts/build.out           # follow progress (ETA is logged per shard)
```

(or run it inside `tmux` / `screen`). If the run stops for any reason, start
the same command again. Embedding shards that are already finished are
skipped, and the run continues where it left off.

To speed up the CPU build: `max_seq_length: 256` makes the longest ~5% of
tickets cheaper (the rest are shorter than 256 tokens anyway). Setting
`cpu_threads` to the number of *physical* cores sometimes helps on servers
with hyper-threading. Both must be decided **before** the build: queries must
use the same settings.

### Model download (BAAI/bge-m3)

The model is downloaded automatically the first time it is needed (about
2.3 GB, into `<project>/.hf_cache/`, git-ignored) and reused after that.
To download and test it before the long build:

```bash
python download_model.py      # prints dimension 1024 and a rough speed when OK
```

This needs HTTPS access to `huggingface.co`. If the server sits behind a
proxy, `export HTTPS_PROXY=http://proxy:port` first. **Without internet
access**, copy a downloaded cache from another machine, for example from the
laptop where TRS_V3.0 already has it:

```bash
# on the laptop (Git Bash): copy the cached model to the server
scp -r "/d/TRS_V3.0/.hf_cache/hub/models--BAAI--bge-m3" user@server:~/go-care-ai/.hf_cache/hub/
# on the server: use only the local copy
export HF_HUB_OFFLINE=1
python download_model.py
```

### Streamlit UI

```bash
export GOCARE_LLM_TOKEN='...'          # only needed for the AI advisor (or put it in .env)
nohup streamlit run app.py --server.address 0.0.0.0 --server.port 8501 > artifacts/ui.out 2>&1 &
```

Open `http://<server>:8501`. If the port is not reachable, tunnel it:
`ssh -L 8501:localhost:8501 user@server`, then open `http://localhost:8501`. The
index and model are loaded once, when the first page is opened (~30 s on CPU).

* **Form**: title, description, optional "already done" note, and the metadata
  fields as dropdowns filled with the values that occur in the history
  (`(unknown)` = do not filter on this field).
* **Sidebar**: number of tickets, minimum similarity, strict/relaxed filters,
  AI advisor status and auto-run toggle, index version.
* **Results**: status, filters applied/relaxed/skipped, warnings, then each
  similar ticket with its metadata, full description and **all its articles in
  full**, plus an overall article ranking and a JSON download.
* **AI recommendation**: button (or auto-run) that sends everything to the LLM.
  It shows the assessment, likely cause, step(s) with source tickets, the
  escalation condition, and the exact prompt that was sent.

### All commands

| Command | What it does |
|---|---|
| `python build_index.py` | Full pipeline: all four stages |
| `python build_index.py --stage clean` | Only clean the CSV and write the report (no embedding). Useful to check the data locally |
| `python build_index.py --stage embed --stage index` | Embed and index an already cleaned dataset |
| `python build_index.py --stage articles` | Build or refresh the article store (e.g. when the article CSV arrives later) |
| `python update_index.py` | Incremental update from the configured CSVs |
| `python update_index.py --tickets new.csv` | Incremental update from a delta file |
| `python update_index.py --snapshot` | Input is a complete export, so tickets missing from it are removed |
| `python update_index.py --full` | Force a full rebuild |
| `python main.py --ticket t.json` | Recommendation for a ticket in a JSON file |
| `python main.py --title "..." --body "..." --IspId VDF --TechnologyType Gpon` | Recommendation from the command line |
| `python main.py --ai --progress "Modem al herstart"` | Also ask the AI advisor |
| `streamlit run app.py` | The agent UI |

All scripts accept `--config path/to/config.yaml` (or the `GOCARE_CONFIG` environment variable).

---

## 2. What the data looks like (verified on `Tickets_oct26.csv`)

| Finding | Consequence in the code |
|---|---|
| The ticket CSV has **no header row**; 53 columns in the same order as the Go-Care `Ticket` table | `tickets.columns` in config.yaml lists the order; `has_header: auto` detects headers in other exports |
| 921,410 tickets, all with a valid `TicketNumber` (`T##########`), no duplicates | De-duplication still runs (keeps the latest `LastStateChangeDate`) |
| `State`: 909,635 `Closed`, the rest Open / Pending / New / Reopened / Draft | Only `Closed` tickets with a `CloseDate` enter the index (`cleaning.eligibility`) |
| **There is no `Vdf` column.** `VDF` is a *value* of `IspId` (99.7% of tickets; others: SMP, B2B, ENR, ENT) | An incoming `Vdf` field is mapped to `IspId` (`filters.input_aliases`) with a warning. `NetworkOperator` (WBA, REG, OPF, DFN, EFR, VWT, GPT, …) is a separate field |
| `TechnologyType` is empty on 48% of tickets (mostly older ones); `???` means unknown | Unknown values never match a known query value. The filter is relaxed early if it leaves too few candidates |
| `CallReasonId` is empty on 17% of tickets | Same rule |
| `RootCauseId` / `SolutionId` are missing only on unclosed tickets | Not required for the incoming ticket; used as a filter only when it is known |
| Article export (`Id, Title, ArticleTypeId, CreatedBy, CreateDate, BodyText, RowVersion, TicketNumber, …`) has a header; one ticket has many articles | Article `Id` is the stable key; the ticket link is `TicketNumber` |

Column names are mapped to canonical names (`Title` → `Title_ticket` / `Title_article`,
`BodyText` → `BodyText_ticket` / `BodyText_article`) through `aliases` in config.yaml.
A merged export that already uses `Title_ticket`, `Title_article` etc. works too.
Missing **required** columns stop the run with an error. Missing optional columns
are logged and listed in the report.

---

## 3. Architecture

| File | Responsibility |
|---|---|
| `config.yaml` | All settings: paths, column mappings, cleaning rules, model, filters, thresholds, batch sizes |
| `config.py` | Loads the settings, artifact names, structured logging |
| `data_loader.py` | Finds CSV files, detects header / schema, maps aliases, reads in chunks (only the needed columns) |
| `data_cleaner.py` | Eligibility rules, validation, de-duplication, cleaning report |
| `ticket_preprocessor.py` | Text cleaning and the canonical `ticket_text` (same code for history and queries) |
| `embedding_service.py` | Loads bge-m3 once, produces unit-length vectors, validates them, resumable batch embedding |
| `vector_store.py` | Embedding shards, FAISS index, row alignment, exact cosine search |
| `metadata_filter.py` | Turns the new ticket's metadata into filter masks (strict → relaxed levels) |
| `article_retriever.py` | Fetches linked articles and ranks them |
| `recommendation_service.py` | Runs the whole query pipeline; `find_similar_tickets`, `get_relevant_articles`, `recommend_resolution` |
| `build_index.py` | Initial build (one command) |
| `update_index.py` | Incremental updates |
| `main.py` | Demo / CLI for a new ticket |
| `llm_advisor.py` | AI advisor: builds the prompt from all tickets + articles, calls the LLM, validates sources |
| `app.py` | Streamlit UI (`.streamlit/config.toml` holds the server settings) |
| `storage_utils.py` | Atomic file writes |

### Artifacts (`artifacts/`)

| File | Content |
|---|---|
| `cleaned_tickets.parquet` | Cleaned eligible tickets incl. `ticket_text` and `text_hash`, sorted by TicketNumber |
| `cleaning_report.json` | Input / output rows, duplicates, missing values per column, excluded rows per reason (with example ids), rules used |
| `embeddings/part-*.npy` + `shards.json` | Embedding matrix (float16 by default), split into shards |
| `ticket_index.faiss` | FAISS inner-product index; vector id *i* = embedding row *i* |
| `ticket_metadata.parquet` | One row per vector: `vector_id`, `TicketNumber`, filter fields, `text_hash`, `is_active` |
| `index_manifest.json` | Index version, dataset version/hash, row counts, model name/revision/dimension, preprocessing version, FAISS settings, source files, history |
| `cleaned_articles.parquet` | Articles sorted by TicketNumber (works as the ticket → article index) |
| `article_report.json` | Article cleaning counts and new/updated/removed articles since the previous build |
| `logs/gocare.log` | Structured JSON logs (rotated) |

**Row alignment:** row *i* of the embeddings, FAISS id *i* and `ticket_metadata.vector_id == i`
are always the same ticket. Rows are only appended, never deleted. A changed or
removed ticket gets `is_active = False`. At startup the service checks that the
three have the same length and stops with a clear error if they don't.

The embeddings are stored as shards rather than one `ticket_embeddings.npy`.
This lets the multi-hour first build resume after an interruption, and lets
updates append vectors without rewriting gigabytes of data.

---

## 4. How filtering, cosine similarity and articles work together

### 4.1 `ticket_text`
```
Title: {cleaned title}
Description: {cleaned description}
```
Cleaning repairs broken characters and removes HTML, markdown, control
characters, the "Imported ticket of type 1 …" boilerplate, URLs, e-mail
addresses and phone numbers. It keeps the Dutch text, error and progress
codes and technical terms. Ticket IDs and other administrative fields are
**not** embedded. The same function prepares incoming tickets.

### 4.2 Embeddings and cosine similarity
`BAAI/bge-m3` (1024 dims, multilingual) encodes `ticket_text`. Every vector is
normalised to length 1, so

```
similarity = historical_embedding · new_ticket_embedding   (= cosine similarity)
```

`similarity_percent` is cosine × 100. It is **not** a probability or a
confidence that the old resolution will work.

At startup the service re-embeds a fixed probe sentence and compares it with
the vector stored at build time. If they differ, the query model is not the
index model and the service refuses to answer (`embedding.strict_model_check`).

### 4.3 Metadata filtering (before ranking)
1. Always excluded: the incoming ticket itself, and replaced or removed ticket versions.
2. Only resolved tickets are in the index at all (cleaning rules).
3. For each field in `filters.fields` (`IspId, TicketTypeId, CallReasonId,
   TechnologyType, NetworkOperator, RootCauseId`) that the **incoming ticket
   knows**, only historical tickets with exactly that value stay. Unknown
   values are skipped (`filters_skipped`), never guessed. A historical ticket
   with an unknown value never matches a known one.
4. `mode: relaxed`: if a level gives fewer than `top_k` results above
   `min_similarity`, the next field in `relax_order` is dropped (RootCauseId →
   NetworkOperator → TechnologyType → TicketTypeId → CallReasonId) and the search
   repeats. `IspId` is never relaxed. `mode: strict` never relaxes.
5. With `prefer_stricter_matches: true` (default), results that match more
   filters rank before results found only after relaxing. Once the list is
   full, relaxing stops. With `false`, everything is ranked by similarity alone,
   and relaxing continues until a level has at least `min_candidates`
   candidates. Each ticket shows its `matched_filters`.

Every step is recorded in `filter_trace`: the filters used, candidate counts,
search method and how many tickets each level added.

### 4.4 Ranking
* At most `exact_search_max_candidates` (100k) candidates: cosine is computed for **every** candidate (exact).
* More candidates: FAISS searches *inside* the filter mask (bitmap selector)
  for `top_k × faiss_overfetch` ids. Their exact cosine is then recomputed
  from the stored vectors. The FAISS result is never the final ranking.
* Results below `retrieval.min_similarity` are dropped, even if fewer than 5
  remain. `retrieval_status` is then `partial` or `no_results`, with an explanation in `warnings`.

### 4.5 Articles
**Articles are not embedded.** Only tickets are embedded and searched.
Articles are stored in `cleaned_articles.parquet`, sorted by `TicketNumber`, so
all articles of a ticket can be looked up directly. For the selected tickets,
every article with the same `TicketNumber` is fetched and de-duplicated by `Id`:

* each ticket carries **all** its `articles` with the full `BodyText_article`,
  in the order they were written (the resolution sequence);
* `article_ranking` lists every article (without body), most similar source
  ticket first, with `source_ticket_similarity`.

`article_retrieval.max_articles_per_ticket: 0` returns all articles. If a
ticket has no article, this is stated explicitly.

### 4.6 AI advisor (`llm_advisor.py`, section `llm:` in config.yaml)
This follows the TRS_V3.0 integration (same OpenAI-compatible gateway, model
and grounding rules), **without the summarisation step**:

* **What is sent:** the new ticket (title, description, known metadata, and the
  optional "already done" note), and for each of the top `max_tickets` (5)
  tickets: its problem (title + full description + metadata) and **every
  linked article in full**, in the order the articles were recorded.
  Nothing is summarised, compressed or dropped. `max_ticket_body_chars` and
  `max_article_chars` are 0 (no limit). Set them only if the model's context
  window is ever exceeded.
* **Privacy:** e-mail addresses, phone numbers and URLs are masked before text
  leaves the server (`mask_personal_data: true`).
* **Grounding:** the system prompt only allows actions that appear in the
  articles of a historical ticket whose problem genuinely matches. If none
  match, the answer is an explicit "no matching history" with a reason per
  ticket. The code then checks every cited ticket number against the tickets
  that were sent, drops unknown sources, and removes steps without a valid source.
* **Output:** `assessment`, `likely_cause`, `steps` (action, why, sources),
  `escalate_if`, `confidence`, plus model, latency, token usage and the exact prompt.
  `response_mode: next_step` (default, as in TRS_V3.0) returns only the single
  next action. `full_plan` returns an ordered plan of up to `max_plan_steps` steps.
* **Gateway:** `base_url` + `model`; the token comes from the environment variable
  named in `token_env` (default `GOCARE_LLM_TOKEN`) or from a git-ignored `.env`
  file. Structured output (`json_schema`) is tried first. If the gateway
  rejects it (HTTP 400), the call is retried in plain JSON mode.

---

## 5. Input and output

**Input** (`sample_ticket.json`); only a title or description is required:
```json
{
  "TicketNumber": "NEW-001",
  "Title_ticket": "Nazorg - Geen Licht",
  "BodyText_ticket": "Odido Fiber installatie. Geen licht op de FTU na installatie, pop niet toegankelijk.",
  "IspId": "VDF",
  "TicketTypeId": "1",
  "CallReasonId": "46",
  "RootCauseId": null,
  "TechnologyType": "Gpon",
  "NetworkOperator": "REG"
}
```
IDs are compared as strings exactly as they appear in the source (`"1"`, `"46"`, `"VDF"`).

**Output** (shape; ids and scores are placeholders):
```json
{
  "query_ticket": "NEW-001",
  "retrieval_status": "success",
  "similarity_metric": "similarity_score = cosine similarity ...",
  "metadata_filters_applied": ["IspId", "TicketTypeId", "CallReasonId", "TechnologyType"],
  "filters_relaxed": ["NetworkOperator"],
  "filters_skipped": {"RootCauseId": "unknown in incoming ticket"},
  "similar_tickets": [
    {
      "TicketNumber": "T25...", "Title_ticket": "...", "BodyText_ticket": "...",
      "TicketTypeId": "1", "IspId": "VDF", "NetworkOperator": "REG", "CallReasonId": "46",
      "RootCauseId": "7", "TechnologyType": "Gpon", "SolutionId": "11",
      "similarity_score": 0.8312, "similarity_percent": 83.12,
      "matched_filters": ["IspId", "TicketTypeId", "CallReasonId", "TechnologyType"],
      "articles": [
        {"Id": "123", "Title_article": "...", "ArticleTypeId": "1", "BodyText_article": "full text ...",
         "TicketNumber": "T25...", "CreateDate": "2025-...", "source_ticket_similarity": 0.8312}
      ]
    }
  ],
  "article_ranking": [{"Id": "123", "TicketNumber": "T25...", "Title_article": "...", "source_ticket_similarity": 0.8312}],
  "filter_trace": [{"filters": ["..."], "relaxed": [], "candidates": 12, "search_method": "exact", "tickets_added": 3}],
  "warnings": [],
  "index": {"index_version": 1, "dataset_version": "20261029T...", "model": "BAAI/bge-m3"}
}
```
Up to `top_k` tickets are returned, not always exactly five.

From Python (e.g. a Go-Care integration or an API wrapper):
```python
from recommendation_service import RecommendationService
service = RecommendationService()            # load once at startup
response = service.recommend_resolution(new_ticket, top_k=5)
```

---

## 6. Incremental updates

`update_index.py` cleans the new input with the same rules, then per ticket:

| Situation | Action |
|---|---|
| new TicketNumber | embed and append |
| `ticket_text` changed (`text_hash` differs) | old vector inactive, new vector appended |
| only metadata changed | metadata updated in place, no embedding |
| ticket in the input but no longer eligible (e.g. reopened) | inactive |
| `--snapshot` and ticket missing from the export | inactive |

New vectors go into a new shard. The FAISS index, metadata, cleaned tickets
and manifest are written to temporary files and moved into place together,
so an interrupted update leaves the previous index intact. The manifest's
`index_version` goes up and a history entry is added. A **full rebuild runs
automatically** when the model, the `text` settings
(`ticket_preprocessor.PREPROCESSING_VERSION`) or the storage dtype changed.
For IVF indexes, a warning suggests a rebuild once 30% more vectors were
added than the index was trained on. Articles are refreshed after each update.

---

## 7. Performance notes (large CSVs)

* Only the ~20 needed columns are read, in chunks of `csv.chunk_size` rows.
* Text cleaning is per chunk. Expect a few minutes for 900k tickets.
* Embedding (CPU): batches of `embedding.batch_size` texts, length-sorted so
  there is little padding, using all cores. Shards of `shard_rows` make the
  long run resumable.
* Query time on CPU: one embedding (the new ticket) plus the vector search
  and a Parquet lookup of the articles, typically well under a second after
  the model is loaded. Articles are never embedded.
* Storage: 909k × 1024 × float16 ≈ 1.9 GB on disk, memory-mapped at query time.
* FAISS: an exact `Flat` index up to 1.5M vectors (~3.8 GB RAM as float32). Above
  that, `IVF` (`ivf_flat`, or `ivf_sq8` for 4× less RAM) with `ivf_nprobe` to trade speed against recall.
* Metadata filters are integer-coded numpy masks: about 1 ms per field for 1M rows.
* Ticket text and articles are read from Parquet with TicketNumber filters
  (sorted files + small row groups), so only a few row groups are read per query.
* The model, index and metadata are loaded once per process. Reuse
  `RecommendationService` instead of creating it per request.

---

## 8. Troubleshooting

| Message / symptom | Fix |
|---|---|
| `No tickets CSV files match ...` | Check `paths.tickets_csv` and that the data is in `Raw Data/` on this machine |
| `no header row and N fields per row, but config lists M columns` | The export layout changed. Update `tickets.columns` (or add a header row) |
| `required ticket columns missing` | Add the source column name to `tickets.aliases` |
| `Embeddings do not match cleaned_tickets.parquet` | Cleaning was re-run after embedding: `python build_index.py --stage embed --stage index` |
| `Artifacts are out of sync` | `python update_index.py --full` |
| `Query model does not produce the same vectors as the index` | Config model ≠ build model, or the model files changed. Fix `embedding.model_name` or rebuild |
| `The 'text' settings changed since the index was built` | Rebuild, or restore the previous `text` settings |
| Model download fails on the server | Check access to huggingface.co (or set `HTTPS_PROXY`), or copy the cache as described under "Model download" and set `HF_HUB_OFFLINE=1` |
| Embedding is slow on the CPU server | Expected for bge-m3; follow the ETA in the log. Keep it running with nohup/tmux, it resumes after a restart |
| UI: "The recommendation index is not ready" | Run `python build_index.py` first (or point `GOCARE_CONFIG` to the right config) |
| UI: "No API token" | `export GOCARE_LLM_TOKEN=...` (or add it to `.env`) **before** starting streamlit |
| AI advisor: HTTP 401/403 | Wrong or expired token, or `auth_scheme` does not match the gateway |
| AI advisor: "answer was cut off" | Raise `llm.max_tokens` |
| AI advisor: context length error | Set `llm.max_article_chars` / `max_ticket_body_chars`, or lower `llm.max_tickets` |
| Many `partial` results | Lower `retrieval.min_similarity`, or check `filter_trace` for a filter that removes too much |
| Article stage skipped | No file matches `paths.articles_csv`. Add it and run `python build_index.py --stage articles` |

Check `artifacts/cleaning_report.json` first when the ticket count looks wrong.
It lists every exclusion reason with example TicketNumbers.
