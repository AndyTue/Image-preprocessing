"""Metadata forensics — standalone POC for document fraud detection."""
from analyzer.classifier import Confidence, MetadataReport, classify
from analyzer.extractor import MetadataExtractor, MetadataSnapshot
from analyzer.rules import Flag, evaluate

__all__ = [
    "MetadataExtractor",
    "MetadataSnapshot",
    "Flag",
    "evaluate",
    "MetadataReport",
    "Confidence",
    "classify",
]
