"""Submit a new ticket and print the recommendation as JSON.

    python main.py                                   # uses sample_ticket.json
    python main.py --ticket my_ticket.json
    python main.py --title "Geen internet" --body "Klant heeft geen verbinding" --IspId VDF --TechnologyType Gpon
    python main.py --ticket my_ticket.json --output result.json
    python main.py --ai --progress "Modem al herstart"   # also ask the AI advisor

From Python (e.g. the Go-Care integration):

    from recommendation_service import recommend_resolution
    response = recommend_resolution({"Title_ticket": "...", "BodyText_ticket": "...", "IspId": "VDF"})
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import PROJECT_ROOT, load_config, setup_logging
from llm_advisor import LLMError, ResolutionAdvisor
from recommendation_service import RecommendationService

METADATA_ARGS = ["TicketNumber", "IspId", "TicketTypeId", "CallReasonId", "RootCauseId", "TechnologyType",
                 "NetworkOperator"]


def read_ticket(args: argparse.Namespace) -> dict:
    if args.title or args.body:
        ticket = {"Title_ticket": args.title, "BodyText_ticket": args.body}
    else:
        path = args.ticket or PROJECT_ROOT / "sample_ticket.json"
        ticket = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in METADATA_ARGS:
        if getattr(args, name) is not None:
            ticket[name] = getattr(args, name)
    return ticket


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recommend resolutions for a new Go-Care ticket.")
    parser.add_argument("--ticket", type=Path, help="JSON file with the new ticket")
    parser.add_argument("--title", help="Ticket title (instead of --ticket)")
    parser.add_argument("--body", help="Ticket description (instead of --ticket)")
    for name in METADATA_ARGS:
        parser.add_argument(f"--{name}")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--ai", action="store_true", help="Also get the AI recommendation (needs llm settings)")
    parser.add_argument("--progress", default="", help="What was already done on the ticket (for --ai)")
    parser.add_argument("--output", type=Path, help="Write the JSON response to this file")
    parser.add_argument("--config", help="Path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)
    service = RecommendationService(cfg)  # loads the stored index once
    ticket = read_ticket(args)
    response = service.recommend_resolution(ticket, top_k=args.top_k)
    if args.ai:
        try:
            advice = ResolutionAdvisor(cfg).advise(ticket, response, args.progress).as_dict()
            response["ai_recommendation"] = {k: v for k, v in advice.items() if k not in ("system_prompt", "prompt")}
        except LLMError as exc:
            response["ai_recommendation"] = {"error": str(exc)}

    text = json.dumps(response, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Response written to {args.output}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    return 0 if response["retrieval_status"] != "invalid_input" else 2


if __name__ == "__main__":
    sys.exit(main())
