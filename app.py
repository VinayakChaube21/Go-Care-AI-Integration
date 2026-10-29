"""Streamlit UI for support agents.

    streamlit run app.py --server.address 0.0.0.0 --server.port 8501

Enter a new ticket, get the most similar resolved tickets with all their
articles, and (optionally) an AI recommendation grounded in those articles.
The index and the embedding model are loaded once per server process.
"""
from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st

from config import load_config, setup_logging
from llm_advisor import LLMError, ResolutionAdvisor
from recommendation_service import RecommendationService

UNKNOWN = "(unknown)"
FILTER_FIELDS = ["IspId", "TicketTypeId", "CallReasonId", "TechnologyType", "NetworkOperator", "RootCauseId"]
STATUS_STYLE = {"success": st.success, "partial": st.warning, "no_results": st.error, "invalid_input": st.error}
CONFIDENCE_ICON = {"high": "🟢", "medium": "🟡", "low": "🔴"}


# -----------------------------------------------------------------------------
# Cached resources (shared by all sessions)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the index and the embedding model ...")
def load_service() -> RecommendationService:
    cfg = load_config()
    setup_logging(cfg)
    return RecommendationService(cfg)


@st.cache_resource
def load_advisor(_service: RecommendationService) -> ResolutionAdvisor:
    return ResolutionAdvisor(_service.cfg)


@st.cache_data(show_spinner=False)
def field_options(_service: RecommendationService) -> dict[str, list[str]]:
    """Values that occur in the historical data, most frequent first."""
    active = _service.store.metadata[_service.store.metadata["is_active"]]
    unknown = _service.filter.unknown
    options = {}
    for name in FILTER_FIELDS:
        if name in active.columns:
            counts = active[name].dropna().astype(str).value_counts()
            options[name] = [v for v in counts.index if v.strip().lower() not in unknown][:300]
    return options


def md(text) -> str:
    """Escape markdown so ticket text is shown literally."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~])", r"\\\1", str(text or ""))


# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Go-Care AI Recommendation", page_icon="🛠️", layout="wide")

try:
    service = load_service()
except (FileNotFoundError, RuntimeError) as exc:
    st.error(f"The recommendation index is not ready: {exc}")
    st.code("python build_index.py", language="bash")
    st.stop()

cfg = service.cfg
advisor = load_advisor(service)
llm_ready, llm_reason = advisor.available()
options = field_options(service)
manifest = service.store.manifest

st.title(cfg.get("ui", {}).get("title", "Go-Care AI Ticket Recommendation"))

with st.sidebar:
    st.header("Search settings")
    top_k = st.slider("Similar tickets", 1, 10, int(cfg["retrieval"]["top_k"]))
    min_similarity = st.slider("Minimum similarity (cosine)", 0.0, 1.0,
                               float(cfg["retrieval"]["min_similarity"]), 0.01)
    filter_mode = st.radio("Metadata filters", ["relaxed", "strict"],
                           index=0 if cfg["filters"].get("mode", "relaxed") == "relaxed" else 1,
                           help="relaxed: optional filters are dropped one by one when too few tickets match.")
    st.header("AI advisor")
    if llm_ready:
        st.caption(f"Model: `{cfg['llm']['model']}`")
        auto_run = st.checkbox("Run automatically after each search", value=bool(cfg["llm"].get("auto_run")))
    else:
        st.caption(llm_reason)
        auto_run = False
    st.header("Index")
    st.caption(f"Version {manifest.get('index_version')} · {manifest['rows']['active_tickets']:,} tickets\n\n"
               f"Model `{manifest['model']['model_name']}`\n\n"
               f"Articles: {'available' if service.articles.available else 'not built yet'}")

# ---- input form ----------------------------------------------------------------
with st.form("new_ticket"):
    left, right = st.columns([2, 1])
    with left:
        title = st.text_input("Title *", placeholder="Nazorg - Geen licht")
        description = st.text_area("Description *", height=180,
                                   placeholder="Odido Fiber installatie. Geen licht op de FTU ...")
        progress = st.text_area("Already done on this ticket (optional, used by the AI advisor)", height=80)
    with right:
        ticket_number = st.text_input("Ticket number (optional)", help="Excluded from the results if it is indexed.")
        chosen = {}
        for name in FILTER_FIELDS:
            values = [UNKNOWN] + options.get(name, [])
            default = values.index("VDF") if name == "IspId" and "VDF" in values else 0
            chosen[name] = st.selectbox(name, values, index=default, key=f"field_{name}")
    submitted = st.form_submit_button("Find similar tickets", type="primary", width="stretch")

if submitted:
    if not (title.strip() or description.strip()):
        st.error("Enter a title or a description.")
        st.stop()
    new_ticket = {"TicketNumber": ticket_number.strip() or None, "Title_ticket": title, "BodyText_ticket": description,
                  **{k: (None if v == UNKNOWN else v) for k, v in chosen.items()}}
    with st.spinner("Searching similar resolved tickets ..."):
        response = service.recommend_resolution(new_ticket, top_k=top_k, min_similarity=min_similarity,
                                                filter_mode=filter_mode)
    st.session_state["last"] = {"ticket": new_ticket, "progress": progress, "response": response, "advice": None,
                                "advice_error": None, "auto_pending": auto_run}

last = st.session_state.get("last")
if not last:
    st.info("Enter a ticket and press **Find similar tickets**.")
    st.stop()

response = last["response"]
tickets = response.get("similar_tickets", [])

# ---- retrieval summary -----------------------------------------------------------
STATUS_STYLE.get(response["retrieval_status"], st.info)(
    f"Status: **{response['retrieval_status']}** - {len(tickets)} similar ticket(s)")
trace = response.get("filter_trace") or [{}]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Similar tickets", len(tickets))
m2.metric("Articles", len(response.get("article_ranking", [])))
m3.metric("Candidates compared", f"{trace[-1].get('candidates', 0):,}")
m4.metric("Search time", f"{response.get('elapsed_ms', 0)} ms")
st.caption(f"Filters applied: {', '.join(response.get('metadata_filters_applied') or []) or 'none'} · "
           f"relaxed: {', '.join(response.get('filters_relaxed') or []) or 'none'} · "
           f"skipped: {', '.join(response.get('filters_skipped') or {}) or 'none'}")
for warning in response.get("warnings", []):
    st.warning(warning)
st.caption("Similarity = cosine similarity of the ticket texts × 100. It is not a probability that a resolution works.")

# ---- AI advisor ----------------------------------------------------------------------
st.subheader("🤖 AI recommendation")
if not llm_ready:
    st.caption(llm_reason)
elif not tickets:
    st.caption("No similar tickets, so there is nothing to base a recommendation on.")
else:
    auto_pending = last.pop("auto_pending", False)
    run_now = st.button("Get AI recommendation", type="primary") or auto_pending
    if run_now:
        with st.spinner(f"Sending {len(tickets)} tickets and all their articles to the model ..."):
            try:
                last["advice"], last["advice_error"] = advisor.advise(last["ticket"], response, last["progress"]), None
            except LLMError as exc:
                last["advice"], last["advice_error"] = None, str(exc)
    if last.get("advice_error"):
        st.error(last["advice_error"])
    advice = last.get("advice")
    if advice:
        st.markdown(f"**Confidence:** {CONFIDENCE_ICON.get(advice.confidence, '')} {advice.confidence}")
        st.markdown(f"**Assessment:** {advice.assessment}")
        if advice.likely_cause:
            st.markdown(f"**Likely cause:** {advice.likely_cause}")
        if advice.steps:
            for number, step in enumerate(advice.steps, start=1):
                st.markdown(f"**{number}. {step.action}**  \n{step.why}  \n_Sources: {', '.join(step.sources)}_")
        else:
            st.info("No grounded step: none of the similar tickets matches this problem well enough.")
        if advice.escalate_if:
            st.markdown(f"**Escalate if:** {advice.escalate_if}")
        for warning in advice.warnings:
            st.warning(warning)
        with st.expander(f"What was sent to the model ({len(advice.tickets_sent)} tickets, {advice.articles_sent} "
                         f"articles, {advice.input_tokens:,} tokens in / {advice.output_tokens:,} out, "
                         f"{advice.latency_ms / 1000:.1f}s)"):
            st.code(advice.prompt, language="markdown")

# ---- similar tickets with their articles -----------------------------------------------------
st.subheader("Similar resolved tickets")
for rank, ticket in enumerate(tickets, start=1):
    header = (f"#{rank} · {ticket['TicketNumber']} · {ticket['similarity_percent']}% similar · "
              f"{(ticket.get('Title_ticket') or '')[:90]}")
    with st.expander(header, expanded=rank == 1):
        info, text = st.columns([1, 2])
        with info:
            meta = {k: ticket.get(k) for k in ("TicketTypeId", "IspId", "NetworkOperator", "CallReasonId",
                                                "RootCauseId", "TechnologyType", "SolutionId", "CloseDate")}
            st.dataframe(pd.DataFrame(meta.items(), columns=["Field", "Value"]).astype(str),
                         hide_index=True, width="stretch")
            st.caption(f"Matched filters: {', '.join(ticket.get('matched_filters') or []) or 'none'}")
        with text:
            st.markdown(f"**{md(ticket.get('Title_ticket'))}**")
            st.text(ticket.get("BodyText_ticket") or "")

        articles = ticket.get("articles", [])
        st.markdown(f"**Resolution articles ({len(articles)})**")
        if not articles:
            st.caption(ticket.get("articles_note") or "No linked resolution article found.")
        for article in articles:
            with st.container(border=True):
                st.markdown(f"**{md(article.get('Title_article') or '(no title)')}**  \n"
                            f"Article {article['Id']} · type {article.get('ArticleTypeId')} · "
                            f"{article.get('CreateDate') or ''} · relevance {article.get('relevance_score')}")
                st.text(article.get("BodyText_article") or "")

if response.get("article_ranking"):
    with st.expander("All articles ranked by relevance"):
        st.dataframe(pd.DataFrame(response["article_ranking"]), hide_index=True, width="stretch")

download = dict(response)
if last.get("advice"):
    download["ai_recommendation"] = last["advice"].as_dict()
st.download_button("Download result as JSON", json.dumps(download, ensure_ascii=False, indent=2, default=str),
                   file_name=f"recommendation_{response.get('query_ticket') or 'ticket'}.json",
                   mime="application/json")
