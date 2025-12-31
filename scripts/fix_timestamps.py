#!/usr/bin/env python3
"""
Fix timestamps for pipeline-generated documents.

The bug: ConversationMessage.create() had `timestamp: int = int(time.time())`
as a default parameter, which is evaluated once at module load time.
All documents created after server start got the same frozen timestamp.

The fix: For each conversation, find the max timestamp from conversation-type
documents (which have correct timestamps), then assign sequential timestamps
to pipeline-generated documents based on their sequence_no.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Document types that are pipeline-generated (affected by the bug)
PIPELINE_DOC_TYPES = {
    'step', 'ner', 'analysis', 'motd', 'codex',
    'brainstorm', 'summary', 'reflection', 'journal',
    'daydream', 'pondering', 'report'
}

# Known frozen timestamps (from analysis)
FROZEN_TIMESTAMPS = {1766428733, 1764458146, 1765866678, 1765417451, 1764798281}


def fix_conversation_file(filepath: Path, dry_run: bool = True) -> dict:
    """Fix timestamps in a single conversation file."""

    docs = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                docs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    if not docs:
        return {'file': str(filepath), 'status': 'empty', 'fixed': 0}

    # Separate conversation docs from pipeline docs
    conv_docs = []
    pipeline_docs = []

    for doc in docs:
        doc_type = doc.get('document_type', 'conversation')
        if doc_type in PIPELINE_DOC_TYPES:
            pipeline_docs.append(doc)
        else:
            conv_docs.append(doc)

    if not pipeline_docs:
        return {'file': str(filepath), 'status': 'no_pipeline_docs', 'fixed': 0}

    # Find max timestamp from conversation docs
    max_conv_ts = 0
    for doc in conv_docs:
        ts = doc.get('timestamp', 0)
        if ts > max_conv_ts:
            max_conv_ts = ts

    # If no conversation docs, use the earliest pipeline doc timestamp
    # that's NOT a frozen timestamp
    if max_conv_ts == 0:
        for doc in pipeline_docs:
            ts = doc.get('timestamp', 0)
            if ts > 0 and ts not in FROZEN_TIMESTAMPS:
                if max_conv_ts == 0 or ts < max_conv_ts:
                    max_conv_ts = ts

    # If still no reference, use current time minus some offset
    if max_conv_ts == 0:
        import time
        max_conv_ts = int(time.time()) - 86400  # 1 day ago

    # Sort pipeline docs by (branch, sequence_no) to maintain order
    pipeline_docs.sort(key=lambda d: (d.get('branch', 0), d.get('sequence_no', 0)))

    # Count how many need fixing
    fixed_count = 0

    # Assign new timestamps: start 1 second after max_conv_ts, increment by 1
    next_ts = max_conv_ts + 1

    for doc in pipeline_docs:
        old_ts = doc.get('timestamp', 0)

        # Check if this timestamp needs fixing
        # Fix if: it's a frozen timestamp, or if multiple pipeline docs share the same timestamp
        if old_ts in FROZEN_TIMESTAMPS or old_ts <= max_conv_ts:
            doc['timestamp'] = next_ts
            fixed_count += 1

        next_ts += 1

    if fixed_count == 0:
        return {'file': str(filepath), 'status': 'no_fixes_needed', 'fixed': 0}

    if not dry_run:
        # Write back all docs in original order
        all_docs = conv_docs + pipeline_docs
        all_docs.sort(key=lambda d: (d.get('branch', 0), d.get('sequence_no', 0)))

        with open(filepath, 'w') as f:
            for doc in all_docs:
                f.write(json.dumps(doc) + '\n')

    return {
        'file': str(filepath),
        'status': 'fixed' if not dry_run else 'would_fix',
        'fixed': fixed_count,
        'total_pipeline': len(pipeline_docs),
        'max_conv_ts': max_conv_ts
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fix timestamps in conversation files')
    parser.add_argument('--dir', default='memory/conversations', help='Conversations directory')
    parser.add_argument('--apply', action='store_true', help='Actually apply fixes (default is dry run)')
    parser.add_argument('--file', help='Fix a single file instead of all')
    args = parser.parse_args()

    dry_run = not args.apply

    if dry_run:
        print("=== DRY RUN MODE (use --apply to actually fix) ===\n")
    else:
        print("=== APPLYING FIXES ===\n")

    if args.file:
        files = [Path(args.file)]
    else:
        conv_dir = Path(args.dir)
        files = list(conv_dir.glob('*.jsonl'))

    total_fixed = 0
    files_modified = 0

    for filepath in sorted(files):
        result = fix_conversation_file(filepath, dry_run=dry_run)

        if result['fixed'] > 0:
            files_modified += 1
            total_fixed += result['fixed']
            print(f"{result['status']}: {filepath.name} - {result['fixed']} docs")

    print(f"\n{'Would fix' if dry_run else 'Fixed'} {total_fixed} documents in {files_modified} files")

    if dry_run and total_fixed > 0:
        print("\nRun with --apply to apply fixes")


if __name__ == '__main__':
    main()
