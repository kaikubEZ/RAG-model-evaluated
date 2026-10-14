"""Data generation commands: generate (questions), answer, translate, freeze (writes dataset.csv), and the app."""
import argparse
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from .. import jobs, llm, store


def _confirm_privacy(yes: bool) -> None:
    """Document text goes to the model providers; say where and ask once per command (PRD section 9)."""
    hosts = sorted({urlparse(llm.config(role)["url"]).netloc for role in ("gen", "judge")})
    print(f"Text from your documents will be sent to: {', '.join(hosts)}", flush=True)
    if not yes and input("Continue? [y/N] ").strip().lower() != "y":
        raise SystemExit("Stopped. Nothing was sent.")


def _dataset(arg: str | None) -> str:
    dataset_id = arg or store.newest_unfrozen()
    if not dataset_id:
        raise SystemExit("There is no dataset being built. Run `rageval-datagen generate` first.")
    return dataset_id


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rageval-datagen", description="Generate and freeze a test-case dataset (CSV)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="stage 1: read material, generate and validate English questions")
    g.add_argument("--subject")
    g.add_argument("--dataset")
    g.add_argument("--yes", action="store_true", help="do not ask before sending document text")

    for name, text in (("answer", "draft and verify English answers for the approved questions"),
                       ("translate", "translate approved questions and answers into the extra test languages"),
                       ("freeze", "run the closed-book check, freeze the dataset and write dataset.csv")):
        c = sub.add_parser(name, help=text)
        c.add_argument("--dataset")
        c.add_argument("--yes", action="store_true")

    for name in ("review", "app"):
        sub.add_parser(name, help="open the app")

    a = p.parse_args(argv)
    try:
        if a.cmd == "generate":
            from . import generate
            _confirm_privacy(a.yes)
            print("dataset:", generate.run(subject=a.subject, dataset_id=a.dataset))
        elif a.cmd == "answer":
            from . import answer
            _confirm_privacy(a.yes)
            print(answer.run(_dataset(a.dataset)))
        elif a.cmd == "translate":
            from . import translate
            _confirm_privacy(a.yes)
            print(translate.run(_dataset(a.dataset)))
        elif a.cmd == "freeze":
            from . import freeze
            _confirm_privacy(a.yes)
            m = freeze.run(_dataset(a.dataset))
            print("frozen:", m["dataset_id"], "csv:", store.datasets_dir() / m["dataset_id"] / "dataset.csv")
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
