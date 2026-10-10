#!/usr/bin/env python3
"""Export one immutable titled source as PNG/JPEG on a retained GPU host."""
import argparse
import json
from pathlib import Path
import platform
import subprocess

try:
    from .package_hindi_delivery import render_thumbnail, sha256_file
except ImportError:
    from package_hindi_delivery import render_thumbnail, sha256_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('expected_sha256')
    ap.add_argument('output_dir', type=Path)
    args = ap.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('Image export requires the configured Linux GPU host')
    subprocess.run(['nvidia-smi', '-L'], check=True, capture_output=True)
    if sha256_file(args.source) != args.expected_sha256:
        raise ValueError('Source hash mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = render_thumbnail(args.source, args.output_dir / 'thumb_hi_titled.png',
                              args.output_dir / 'thumb_hi_upload_1280.jpg', 2_000_000)
    report.pop('sameApprovedComposition')
    report.update(sourceSha256=args.expected_sha256, sameSourceComposition=True,
                  wordingApproval='recorded separately', finalVisualApproval=False)
    (args.output_dir / 'export-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
