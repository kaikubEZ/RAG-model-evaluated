"""Evaluation commands: run, score, compare, and the app. Reads datasets/<id>/dataset.csv only."""
import argparse
from urllib.parse import urlparse

from .. import jobs, llm, store


def _confirm_privacy(yes: bool) -> None:
    host = urlparse(llm.config("judge")["url"]).netloc
    print(f"Questions, answers and retrieved text will be sent to the judge model at: {host}", flush=True)
    if not yes and input("Continue? [y/N] ").strip().lower() != "y":
        raise SystemExit("Stopped. Nothing was sent.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rageval-eval", description="Evaluate a RAG endpoint on a dataset CSV")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="ask every question in datasets/<id>/dataset.csv")
    r.add_argument("--dataset")
    r.add_argument("--label", default="")
    r.add_argument("--resume", metavar="RUN_ID")

    s = sub.add_parser("score", help="compute metrics and write the report")
    s.add_argument("--run", required=True)
    s.add_argument("--refresh-judge", action="store_true")
    s.add_argument("--yes", action="store_true")

    c = sub.add_parser("compare", help="paired comparison of two runs on the same dataset")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--primary", default="answer_recall")
    c.add_argument("--only", metavar="KEY=VALUE", help="scenario=same|similar|diff_topic or direction=<q lang>-<doc lang>")

    sub.add_parser("app", help="open the app")

    a = p.parse_args(argv)
    try:
        if a.cmd == "run":
            from . import run
            if not a.resume and not a.dataset:
                raise SystemExit("Give --dataset ID for a new run, or --resume RUN_ID.")
            print("run:", run.run(dataset_id=a.dataset, label=a.label, run_id=a.resume))
        elif a.cmd == "score":
            from . import score
            _confirm_privacy(a.yes)
            result = score.run(a.run, refresh_judge=a.refresh_judge)
            print("report:", store.runs_dir() / a.run / "report.md")
            if a.refresh_judge:
                print("share of judge verdicts that changed:", result.get("flip_rate"))
        elif a.cmd == "compare":
            from . import report
            print(report.compare_markdown(report.compare(a.run_a, a.run_b, primary=a.primary, only=a.only)))
        else:
            from ..cli import open_app
            open_app()
    except jobs.Busy as e:
        raise SystemExit(str(e))
    except jobs.Stopped:
        raise SystemExit("Stopped. Run the same command again to continue where it stopped.")


if __name__ == "__main__":
    main()
