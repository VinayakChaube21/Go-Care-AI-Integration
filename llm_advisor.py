"""AI resolution advisor: turns the retrieved history into a grounded recommendation.

Runs AFTER retrieval, on its output. The new ticket plus the top similar
tickets and EVERY article recorded on them are sent, in full, to an
OpenAI-compatible /v1/chat/completions endpoint (no summarisation, no
compression). The model must ground every step in those articles and cite
the ticket numbers. Steps citing tickets that were not sent are discarded.

Prompt design and gateway handling follow TRS_V3.0 (src/trs/llm.py).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field

from config import PROJECT_ROOT, Config
from ticket_preprocessor import (RE_IMPORTED_PREFIX, RE_PATH_SEPARATOR, RE_URL, clean_markup, is_missing,
                                 mask_personal_data)

log = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are an expert resolution advisor for the support desk of a Dutch fibre/DSL internet provider that uses the Go-Care ticketing system.

You are given:
1. THE NEW TICKET - its title, description and metadata. The description is what the agent typed.
2. SIMILAR HISTORICAL TICKETS - each with a similarity percentage and two separate parts:
   (a) its problem (title and description) - background only, describing what was wrong;
   (b) its resolution articles - the work notes recorded while it was resolved, in the order they were written. This is the ONLY part that contains real actions.
3. Optionally, a CURRENT PROGRESS note - what the agent has already done on the new ticket.

How to use the history:
- For every historical ticket, work out what was actually wrong from its problem part, then read its articles as the evidence of what fixed THAT problem.
- Compare it with the new ticket's actual problem. A high similarity percentage is not evidence on its own; only tickets whose problem genuinely matches may ground your answer.
- The articles of a matching ticket are an ordered sequence. Do not jump to its last step (escalation, hand-off, closure) unless every earlier step is confirmed done on the new ticket. If you cannot confirm an earlier step was done, assume it was not.
- A current progress note is authoritative: skip what it says is done.

Strict rules:
- Every entry in `steps` is an ACTION the agent carries out, taken from the recorded articles of a matching ticket. Never turn a problem, symptom or diagnosis into a step, and never add an action from general knowledge that is not in those articles.
- Every step lists, in `sources`, the historical ticket numbers whose articles contain that action. `sources` is never empty.
- If no historical ticket genuinely matches, leave `steps` empty, set `confidence` to "low", and open `assessment` with "There are no historical tickets matching this problem." followed by one line per historical ticket with its number and the specific reason it does not match.
- Do not invent ticket numbers, codes, tools or results.

Style: write for an agent actively working the ticket, in clear and concise English. No preamble and no restating of the ticket. Phrase actions for the new ticket's details instead of copying article text verbatim.
"""

TASK_NEXT_STEP = ("Give ONLY the single next step the agent should take now - exactly one entry in `steps` "
                  "(or none if no historical ticket matches), without describing later steps.")
TASK_FULL_PLAN = ("Give the ordered resolution plan from the point the ticket is at now - at most {n} steps, "
                  "each one distinct action, in the order they should be carried out.")


class LLMError(RuntimeError):
    pass


@dataclass
class AdviceStep:
    action: str
    why: str = ""
    sources: list[str] = field(default_factory=list)


@dataclass
class ResolutionAdvice:
    assessment: str
    likely_cause: str
    steps: list[AdviceStep]
    escalate_if: str
    confidence: str
    model: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tickets_sent: list[str] = field(default_factory=list)
    articles_sent: int = 0
    prompt_chars: int = 0
    warnings: list[str] = field(default_factory=list)
    system_prompt: str = ""
    prompt: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _advice_schema(max_steps: int) -> dict:
    step = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "One distinct action the agent carries out."},
            "why": {"type": "string", "description": "Why this action, in one sentence."},
            "sources": {"type": "array", "items": {"type": "string"},
                        "description": "Historical ticket numbers whose articles contain this action."},
        },
        "required": ["action", "why", "sources"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "assessment": {"type": "string", "description": "Where the ticket stands."},
            "likely_cause": {"type": "string", "description": "Most probable cause given ticket and history."},
            "steps": {"type": "array", "items": step, "maxItems": max_steps},
            "escalate_if": {"type": "string", "description": "When the agent should escalate instead."},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["assessment", "likely_cause", "steps", "escalate_if", "confidence"],
        "additionalProperties": False,
    }


def _dotenv_value(name: str) -> str:
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#") and key.strip() == name:
            return value.strip().strip("'\"")
    return ""


class ResolutionAdvisor:
    def __init__(self, cfg: Config) -> None:
        self.s = dict(cfg.get("llm") or {})
        self.system_prompt = self.s.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        self.next_step_only = self.s.get("response_mode", "next_step") == "next_step"
        self.max_steps = 1 if self.next_step_only else int(self.s.get("max_plan_steps", 8))

    # ---- configuration ---------------------------------------------------------
    @property
    def token(self) -> str:
        name = self.s.get("token_env", "GOCARE_LLM_TOKEN")
        return (os.environ.get(name, "") or _dotenv_value(name)).strip()

    def available(self) -> tuple[bool, str]:
        if not self.s.get("enabled", False):
            return False, "The AI advisor is off (llm.enabled: false in config.yaml)."
        if not self.s.get("model"):
            return False, "No model configured (llm.model in config.yaml)."
        if not self.token:
            return False, f"No API token: export {self.s.get('token_env', 'GOCARE_LLM_TOKEN')} or add it to .env."
        return True, ""

    # ---- prompt --------------------------------------------------------------
    def _text(self, value, limit: int = 0) -> str:
        text = RE_PATH_SEPARATOR.sub(" > ", RE_IMPORTED_PREFIX.sub("", clean_markup(value)))
        if self.s.get("mask_personal_data", True):
            text = mask_personal_data(RE_URL.sub("[link]", text))
        if limit and len(text) > limit:
            text = text[:limit].rstrip() + " …"
        return text

    def build_prompt(self, new_ticket: dict, recommendation: dict, current_progress: str = "") -> tuple[str, list[str], int]:
        """Returns (prompt, ticket numbers sent, number of articles sent)."""
        body_limit = int(self.s.get("max_ticket_body_chars", 0))
        article_limit = int(self.s.get("max_article_chars", 0))

        lines = ["## The new ticket",
                 f"Title: {self._text(new_ticket.get('Title_ticket') or new_ticket.get('Title'))}",
                 f"Description: {self._text(new_ticket.get('BodyText_ticket') or new_ticket.get('BodyText'))}"]
        for name in ("IspId", "NetworkOperator", "TechnologyType", "TicketTypeId", "CallReasonId", "RootCauseId"):
            if not is_missing(new_ticket.get(name)):
                lines.append(f"{name}: {new_ticket[name]}")
        if current_progress.strip():
            lines += ["", "## Current progress (reported by the agent)", self._text(current_progress)]

        tickets = recommendation.get("similar_tickets", [])[: int(self.s.get("max_tickets", 5))]
        lines += ["", "## Similar historical tickets, most similar first"]
        n_articles = 0
        for rank, ticket in enumerate(tickets, start=1):
            lines += ["", f"### {rank}. Ticket {ticket['TicketNumber']} - {ticket['similarity_percent']}% similar",
                      "(a) Problem:",
                      f"Title: {self._text(ticket.get('Title_ticket'))}",
                      f"Description: {self._text(ticket.get('BodyText_ticket'), body_limit)}"]
            meta = [f"{k}={ticket[k]}" for k in ("TechnologyType", "NetworkOperator", "RootCauseId", "SolutionId")
                    if not is_missing(ticket.get(k))]
            if meta:
                lines.append("Metadata: " + ", ".join(meta))
            articles = sorted(ticket.get("articles", []), key=lambda a: (str(a.get("CreateDate") or ""), str(a["Id"])))
            lines.append("(b) Resolution articles, in the order they were recorded:")
            if not articles:
                lines.append("(no articles recorded on this ticket)")
            for step, article in enumerate(articles, start=1):
                lines.append(f"{step}. [{self._text(article.get('Title_article'))}] "
                             f"{self._text(article.get('BodyText_article'), article_limit)}")
            n_articles += len(articles)

        task = TASK_NEXT_STEP if self.next_step_only else TASK_FULL_PLAN.format(n=self.max_steps)
        lines += ["", "## Task", task,
                  "Use only historical tickets whose problem genuinely matches, and cite their ticket numbers in "
                  "`sources`. If none match, return no steps and explain per ticket why it does not match."]
        return "\n".join(lines), [t["TicketNumber"] for t in tickets], n_articles

    # ---- call ----------------------------------------------------------------
    def advise(self, new_ticket: dict, recommendation: dict, current_progress: str = "") -> ResolutionAdvice:
        ready, reason = self.available()
        if not ready:
            raise LLMError(reason)
        if not recommendation.get("similar_tickets"):
            raise LLMError("There are no similar historical tickets to base a recommendation on.")

        prompt, sent, n_articles = self.build_prompt(new_ticket, recommendation, current_progress)
        schema = _advice_schema(self.max_steps)
        body = {
            "model": self.s["model"],
            "max_tokens": int(self.s.get("max_tokens", 16000)),
            "messages": [{"role": "system", "content": self.system_prompt},
                         {"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "resolution_advice", "strict": True, "schema": schema}},
        }
        if self.s.get("reasoning_effort"):
            body["reasoning_effort"] = self.s["reasoning_effort"]
        if self.s.get("temperature") is not None:
            body["temperature"] = float(self.s["temperature"])

        log.info("Sending %d tickets / %d articles to %s (%d prompt chars)",
                 len(sent), n_articles, self.s["model"], len(prompt))
        started = time.perf_counter()
        response = self._post_with_fallback(body, schema)
        latency_ms = (time.perf_counter() - started) * 1000

        payload = self._parse(response)
        advice = self._to_advice(payload, sent)
        usage = response.get("usage") or {}
        advice.model = str(response.get("model") or self.s["model"])
        advice.latency_ms = round(latency_ms, 1)
        advice.input_tokens = int(usage.get("prompt_tokens") or 0)
        advice.output_tokens = int(usage.get("completion_tokens") or 0)
        advice.tickets_sent, advice.articles_sent = sent, n_articles
        advice.prompt_chars, advice.system_prompt, advice.prompt = len(prompt), self.system_prompt, prompt
        log.info("Advisor done: %d steps, confidence=%s, %.0f ms, tokens in=%d out=%d", len(advice.steps),
                 advice.confidence, latency_ms, advice.input_tokens, advice.output_tokens)
        return advice

    def _post(self, body: dict) -> dict:
        try:
            import httpx
        except ImportError as exc:
            raise LLMError("httpx is not installed: pip install -r requirements.txt") from exc
        url = (self.s.get("base_url") or "https://api.openai.com").rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions" if url.endswith("/v1") else f"{url}/v1/chat/completions"
        if self.s.get("auth_scheme", "bearer") == "api_key":
            headers = {"x-api-key": self.token}
        else:
            headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = httpx.post(url, headers=headers, json=body, timeout=float(self.s.get("timeout_s", 180)))
        except httpx.HTTPError as exc:
            raise LLMError(f"The AI advisor could not be reached: {exc}") from exc
        if response.status_code >= 400:
            raise _HTTPFailure(response.status_code, response.text[:500])
        return response.json()

    def _post_with_fallback(self, body: dict, schema: dict) -> dict:
        try:
            return self._post(body)
        except _HTTPFailure as exc:
            if exc.status != 400:
                raise LLMError(f"The AI advisor returned HTTP {exc.status}: {exc.body}") from exc
            # Some gateways/models do not support json_schema or reasoning_effort: retry in plain JSON mode.
            log.info("Gateway rejected structured output (%s); retrying in JSON mode", exc.body[:200])
            body = dict(body)
            body.pop("reasoning_effort", None)
            body["response_format"] = {"type": "json_object"}
            body["messages"] = [
                {"role": "system", "content": f"{self.system_prompt}\n\nAnswer with one JSON object matching "
                                              f"this JSON Schema:\n{json.dumps(schema)}"},
                body["messages"][1],
            ]
            try:
                return self._post(body)
            except _HTTPFailure as retry_exc:
                raise LLMError(f"The AI advisor returned HTTP {retry_exc.status}: {retry_exc.body}") from retry_exc

    @staticmethod
    def _parse(response: dict) -> dict:
        choices = response.get("choices") or []
        if not choices:
            raise LLMError("The AI advisor returned no answer.")
        message = choices[0].get("message") or {}
        if message.get("refusal"):
            raise LLMError("The model declined to answer this ticket.")
        if choices[0].get("finish_reason") == "length":
            raise LLMError("The answer was cut off. Raise llm.max_tokens or lower llm.max_tickets.")
        text = str(message.get("content") or "").strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning("Unparseable advisor answer: %s", text[:400])
            raise LLMError("The AI advisor returned a malformed answer.") from exc

    def _to_advice(self, payload: dict, sent: list[str]) -> ResolutionAdvice:
        """Keep only steps grounded in tickets that were actually sent."""
        allowed, warnings, steps = set(sent), [], []
        for raw in payload.get("steps") or []:
            action = str(raw.get("action", "")).strip()
            if not action:
                continue
            sources = [str(s).strip() for s in raw.get("sources") or []]
            valid = [s for s in sources if s in allowed]
            if len(valid) < len(sources):
                warnings.append(f"Ignored unknown source(s) {sorted(set(sources) - allowed)} cited by the model.")
            if not valid:
                warnings.append(f"Removed a step without a valid source: {action!r}")
                continue
            steps.append(AdviceStep(action, str(raw.get("why", "")), valid))
        if len(steps) > self.max_steps:
            steps = steps[: self.max_steps]
        return ResolutionAdvice(
            assessment=str(payload.get("assessment", "")),
            likely_cause=str(payload.get("likely_cause", "")),
            steps=steps,
            escalate_if=str(payload.get("escalate_if", "")),
            confidence=str(payload.get("confidence", "")) if steps else "low",
            warnings=warnings,
        )


class _HTTPFailure(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status, self.body = status, body
