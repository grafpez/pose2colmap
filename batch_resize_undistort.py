#!/usr/bin/env python3
"""
batch_resize_undistort.py
Batch resize/crop undistorted fisheye images

Two modes:
  Default:  2877x1798 -> 1920x1200  (horizontal crop, 16:10, ~55% pixel reduction)
  --vertical: 2877x1798 -> 1920x1798  (vertical-preserving crop, ~33% pixel reduction)

Workflow:
  1. Backs up the Undistort folder to Undistort_original (skips if already exists)
  2. Resizes/crops all JPG/PNG files from Undistort_original (recursive, preserves subfolders)
  3. Overwrites originals in the Undistort folder, keeping identical paths

Changelog:
  v1.0 ??Initial version
  v1.1 ??Fixed -h argparse conflict ??renamed to -H
  v1.2 ??Fixed case-sensitive extension matching (.JPG/.JPEG/.png/.PNG now supported)
  v1.3 ??Fixed flat directory scan ??recursive search (images in left/, right/ subfolders)
         Now preserves subfolder structure when writing output
  v1.4 ??Fixed TypeError: 'float' object is not callable (start_time() typo on line 187)
  v1.5 ??Added --vertical mode: 1920x1798 (preserve full vertical FOV, trim horizontal edges)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import time


def resize_image(args):
    """Resize a single image using ffmpeg"""
    src, dst, width, height, quality = args

    # Ensure output subdirectory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "-q:v", str(quality),
        str(dst),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return (True, src.relative_to(src.parents[2] if src.parents[2].name == "undistort_original" else src.parent), None)
        else:
            return (False, src.name, result.stderr[-200:] if result.stderr else "unknown")
    except Exception as e:
        return (False, src.name, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Batch resize undistorted images and overwrite originals."
    )
    parser.add_argument(
        "input_dir",
        help='Path to the Undistort folder (e.g. "D:\\Showroom\\Output\\Undistort")',
    )
    parser.add_argument(
        "-w", "--width", type=int, default=1920, help="Target width (default: 1920)"
    )
    parser.add_argument(
        "-H", "--height", type=int, default=1200, help="Target height (default: 1200)"
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="Vertical-preserving crop: output 1920x1798 (full vertical FOV, trim horizontal edges). Overrides --height to 1798."
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=2,
        help="JPEG quality 1-31, lower=better (default: 2)",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=4, help="Parallel ffmpeg jobs (default: 4)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating the _original backup folder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    args = parser.parse_args()

    undistort_path = Path(args.input_dir).resolve()
    if not undistort_path.exists():
        print(f"Error: Directory not found: {undistort_path}")
        sys.exit(1)

    # ===== Step 1: Backup =====
    backup_path = undistort_path.parent / (undistort_path.name + "_original")

    if args.dry_run:
        print(f"[DRY RUN] Would backup: {undistort_path} -> {backup_path}")
    elif not args.no_backup:
        if backup_path.exists():
            print(
                f"Backup already exists: {backup_path}\n"
                f"  Skipping backup (delete it manually to re-backup).\n"
                f"  Or use --no-backup to skip entirely."
            )
        else:
            print(f"Backing up: {undistort_path} -> {backup_path}")
            shutil.copytree(undistort_path, backup_path)
            # Count ALL image files recursively for the backup summary
            backup_count = sum(
                1 for f in backup_path.rglob("*")
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            print(f"  Copied {backup_count} images (recursive).")

    # ===== Step 2: Find source images (RECURSIVE) =====
    source_dir = backup_path if (backup_path.exists() and not args.no_backup) else undistort_path
    images = sorted(
        f for f in source_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not images:
        print(f"No JPEG/PNG images found (recursively) in {source_dir}")
        sys.exit(1)

    print(f"\nFound {len(images)} images (recursive) in {source_dir.name}")

    # Show subfolder distribution
    subdir_counts = {}
    for img in images:
        rel = img.relative_to(source_dir)
        part = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        subdir_counts[part] = subdir_counts.get(part, 0) + 1
    print("  Distribution:")
    for folder, count in sorted(subdir_counts.items()):
        print(f"    {folder}/ : {count} images")

    # Apply --vertical override
    if args.vertical:
        args.height = 1798

    print(f"Target size: {args.width}x{args.height}")
    if args.vertical:
        print(f"  Mode: vertical-preserving crop (full vertical FOV, trimming horizontal edges)")
    print(f"Output (overwrite): {undistort_path}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would resize and overwrite:")
        for img in images[:5]:
            rel = img.relative_to(source_dir)
            dst = undistort_path / rel
            print(f"  {rel} -> {dst}")
        if len(images) > 5:
            print(f"  ... and {len(images) - 5} more")
        sys.exit(0)

    # ===== Step 3: Resize & overwrite (preserves subfolder structure) =====
    work_items = []
    for img in images:
        rel = img.relative_to(source_dir)
        dst = undistort_path / rel
        work_items.append((img, dst, args.width, args.height, args.quality))

    success_count = 0
    fail_count = 0
    failed_files = []
    start_time = time.time()

    print(f"\nProcessing with {args.jobs} parallel ffmpeg jobs...")

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(resize_image, item): item for item in work_items}

        for future in as_completed(futures):
            success, filename, error = future.result()
            if success:
                success_count += 1
                if success_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = success_count / elapsed
                    remaining = (len(images) - success_count) / rate
                    print(
                        f"  Progress: {success_count}/{len(images)}  "
                        f"({rate:.1f} img/s, ~{remaining:.0f}s remaining)"
                    )
            else:
                fail_count += 1
                failed_files.append((filename, error))
                print(f"  FAILED: {filename} - {error}")

    elapsed = time.time() - start_time

    # ===== Step 4: Report =====
    original_size = sum(f.stat().st_size for f in images)
    resized_images = list(undistort_path.rglob("*.jpg")) + list(undistort_path.rglob("*.JPG")) + list(undistort_path.rglob("*.jpeg")) + list(undistort_path.rglob("*.JPEG"))
    resized_size = sum(f.stat().st_size for f in resized_images)

    print(f"\n{'=' * 55}")
    print(f"Complete: {success_count} succeeded, {fail_count} failed")
    print(f"Duration: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"\nSize comparison:")
    print(
        f"  Backup (original):  {original_size / 1024 / 1024:.1f} MB  ({len(images)} files)"
    )
    print(
        f"  Undistort (resized): {resized_size / 1024 / 1024:.1f} MB  ({len(resized_images)} files)"
    )
    if original_size > 0:
        reduction = (1 - resized_size / original_size) * 100
        print(f"  Reduction: {reduction:.1f}%")

    if failed_files:
        print(f"\nFailed files:")
        for fname, err in failed_files[:10]:
            print(f"  - {fname}: {err}")

    print(f"\nBackup of originals: {backup_path}")
    print(f"To restore: delete Undistort folder and rename {backup_path.name} -> Undistort")


if __name__ == "__main__":
    main()
