"""Start the BAM web app.

    python run.py
    python run.py --port 8001 --set infer.device=cuda
    python run.py --serve waitress --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import threading
import webbrowser

from config import Config, add_config_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BAM segmentation app")
    add_config_args(parser)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="enable Flask debug mode")
    parser.add_argument(
        "--serve",
        choices=["dev", "waitress"],
        default="dev",
        help="dev server or a production WSGI server",
    )
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config.load(args.overrides)
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if args.debug:
        cfg.debug = True
    cfg.validate()

    cfg.datasets_root.mkdir(parents=True, exist_ok=True)

    from backend.app import create_app

    app = create_app(cfg)

    url = f"http://{'127.0.0.1' if cfg.host == '0.0.0.0' else cfg.host}:{cfg.port}"
    print(f"[bam] serving on {url}")
    print(f"[bam] datasets: {cfg.datasets_root}")

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    if args.serve == "waitress":
        try:
            from waitress import serve
        except ImportError:
            print("[bam] waitress is not installed; falling back to the dev server")
        else:
            serve(app, host=cfg.host, port=cfg.port, threads=8)
            return 0

    app.run(
        host=cfg.host,
        port=cfg.port,
        debug=cfg.debug,
        threaded=True,
        use_reloader=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
