"""Accessors for the objects attached to the Flask app."""

from __future__ import annotations

from flask import current_app, jsonify

from .inference.registry import ModelRegistry
from .services.dataset_index import DatasetIndex


def cfg():
    return current_app.config["BAM"]


def index() -> DatasetIndex:
    return current_app.extensions["bam_index"]


def registry() -> ModelRegistry:
    return current_app.extensions["bam_registry"]


def ok(**payload):
    body = {"status": "success"}
    body.update(payload)
    return jsonify(body)


def fail(message: str, code: int = 400, **payload):
    body = {"status": "error", "message": str(message)}
    body.update(payload)
    return jsonify(body), code


__all__ = ["cfg", "index", "registry", "ok", "fail"]
