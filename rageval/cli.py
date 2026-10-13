"""Command line: one subcommand per step (architecture section 2). The app starts these same commands."""
import argparse
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from . import jobs, llm, store


def _confirm_privacy(yes: bool) -> None:
    """Document text goes to the model providers; say where and ask once per command (PRD section 9)."""
    hosts = sorted({urlparse(llm.config(role)["url"]).netloc for role in ("gen", "judge")})
    print(f"Text from your documents will be sent to: {', '.join(hosts)}", flush=True)
    if not yes and input("Continue? [y/N] ").strip().lower() != "y":
        raise SystemExit("Stopped. Nothing was sent.")


def _dataset(arg: str | None) -> str:
    dataset_id = arg or store.newest_unfrozen()
    if not dataset_id:
        raise SystemExit("There is no dataset being built. Run `rageval generate` first.")
    return dataset_id


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rageval", description="Universal RAG Evaluator")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="read material, generate and validate candidate questions")
    g.add_argument("--subject")
    g.add_argument("--dataset")
    g.add_argument("--yes", action="store_true", help="do not ask before sending document text")

    for name, text in (("translate", "translate approved questions into the extra test languages"),
                       ("freeze", "run the closed-book check and freeze the dataset")):
        c = sub.add_parser(name, help=text)
        c.add_argument("--dataset")
        c.add_argument("--yes", action="store_true")

    r = sub.add_parser("run", help="ask every frozen question")
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

    for name in ("review", "app"):
        sub.add_parser(name, help="open the app")

    a = p.parse_args(argv)
    try:
        if a.cmd == "generate":
            from . import generate
            _confirm_privacy(a.yes)
            print("dataset:", generate.run(subject=a.subject, dataset_id=a.dataset))
        elif a.cmd == "translate":
            from . import translate
            _confirm_privacy(a.yes)
            print(translate.run(_dataset(a.dataset)))
        elif a.cmd == "freeze":
            from . import freeze
            _confirm_privacy(a.yes)
            m = freeze.run(_dataset(a.dataset))
            print("frozen:", m["dataset_id"], m["sha256"][:12])
        elif a.cmd == "run":
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
            app = Path(__file__).with_name("app.py")
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--server.address", "localhost"],
                           cwd=store.ROOT, check=False)
    except jobs.Busy as e:
        raise SystemExit(str(e))
    except jobs.Stopped:
        raise SystemExit("Stopped. Run the same command again to continue where it stopped.")


if __name__ == "__main__":
    main()
