"""`rageval app` opens the one app. The jobs it starts run through rageval-datagen and rageval-eval."""
import subprocess
import sys
from pathlib import Path

from . import store


def open_app() -> None:
    app = Path(__file__).with_name("app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--server.address", "localhost"], cwd=store.ROOT, check=False)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if args not in ([], ["app"]):
        raise SystemExit("Usage: rageval app   (the jobs are run with rageval-datagen and rageval-eval)")
    open_app()


if __name__ == "__main__":
    main()
