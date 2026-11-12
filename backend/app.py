"""Flask application factory.

Templates and static files live under ``frontend/`` so the repository keeps the
requested top-level split (frontend/ and backend/ are siblings).

The app starts even when no checkpoint exists: the model registry falls back to a
mock backend, ``/api/health`` reports ``model: mock``, and the UI shows a warning.
That makes it possible to exercise the whole product before training finishes.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, jsonify, request

from .inference.registry import ModelRegistry
from .services.dataset_index import DatasetIndex

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"


def create_app(config=None, overrides=None) -> Flask:
    from config import Config

    cfg = config or Config.load(overrides)
    cfg.validate()

    app = Flask(
        __name__,
        template_folder=str(FRONTEND / "templates"),
        static_folder=str(FRONTEND / "static"),
        static_url_path="/static",
    )
    app.config["BAM"] = cfg
    app.config["MAX_CONTENT_LENGTH"] = int(cfg.max_upload_mb) * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False
    app.json.sort_keys = False

    logging.basicConfig(
        level=logging.DEBUG if cfg.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app.extensions["bam_index"] = DatasetIndex(cfg.datasets_root)
    app.extensions["bam_registry"] = ModelRegistry(cfg)

    from .routes import datasets as datasets_routes
    from .routes import health as health_routes
    from .routes import magic_brush as magic_brush_routes
    from .routes import masks as masks_routes
    from .routes import pages as pages_routes

    app.register_blueprint(pages_routes.bp)
    app.register_blueprint(datasets_routes.bp)
    app.register_blueprint(masks_routes.bp)
    app.register_blueprint(magic_brush_routes.bp)
    app.register_blueprint(health_routes.bp)

    @app.errorhandler(413)
    def too_large(_error):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"payload exceeds {cfg.max_upload_mb} MB",
                }
            ),
            413,
        )

    def _is_api(path: str) -> bool:
        return path.startswith("/api/") or path.startswith("/magic_brush/")

    @app.errorhandler(404)
    def not_found(_error):
        if _is_api(request.path):
            return jsonify({"status": "error", "message": "not found"}), 404
        return "Not found", 404

    @app.errorhandler(500)
    def server_error(error):
        app.logger.exception("unhandled error: %s", error)
        if _is_api(request.path):
            return jsonify({"status": "error", "message": "internal server error"}), 500
        return "Internal server error", 500

    @app.after_request
    def no_cache_api(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    info = app.extensions["bam_registry"].info()
    print(f"[bam] device={info['device']} backend={info['model']} mock={info['is_mock']}")
    if info.get("error"):
        print(f"[bam] model unavailable: {info['error']}")

    return app


__all__ = ["create_app"]
