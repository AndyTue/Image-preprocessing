from __future__ import annotations

import json
import sys
from dataclasses import asdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

from analyzer.classifier import MetadataReport, classify
from analyzer.extractor import MetadataExtractor, MetadataSnapshot

INPUT: str | list[str] = "../examples/INE.pdf"
SHOW_SNAPSHOT = True


def _print_snapshot(snap: MetadataSnapshot) -> None:
    payload = asdict(snap)
    payload.pop("source", None)
    print("\n  Extracted metadata:")
    print(_indent(json.dumps(payload, indent=2, default=str, ensure_ascii=False), 4))


def _print_report(report: MetadataReport) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {report.source}")
    print("=" * 70)
    print(f"  Format        : {report.format}")
    print(f"  Confidence    : {report.confidence}")
    print(f"  Suspicion     : {report.suspicion_score:.3f}")
    print(f"  Summary       : {report.summary}")
    if not report.flags:
        print("\n  Flags: none")
        return
    print(f"\n  Flags ({len(report.flags)}):")
    for flag in report.flags:
        print(f"    [{flag.severity.upper():<8}] {flag.code:<32} (cat {flag.category})")
        print(f"             {flag.evidence}")


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def main() -> None:
    extractor = MetadataExtractor()
    paths = [INPUT] if isinstance(INPUT, str) else list(INPUT)
    for path in paths:
        snap = extractor.extract(path)
        report = classify(snap)
        _print_report(report)
        if SHOW_SNAPSHOT:
            _print_snapshot(snap)


if __name__ == "__main__":
    main()
