#!/bin/bash
# Move (not copy) output_vla from exa_project1 to exa_data1.
# Runs rsync, verifies via file count + total size match, then removes the
# source only if verification passes. Meant to run inside tmux so it
# survives a Claude Code session ending.
set -e

SRC=/e/project1/reformo/nguyen38/output_vla
DST=/e/data1/datasets/playground/mmlaion/shared/nguyen38/output_vla

echo "=== $(date) Starting rsync $SRC -> $DST ==="
mkdir -p "$DST"
rsync -a --info=progress2 "$SRC/" "$DST/"
echo "=== $(date) rsync finished, verifying ==="

SRC_COUNT=$(find "$SRC" | wc -l)
DST_COUNT=$(find "$DST" | wc -l)
SRC_SIZE=$(du -sb "$SRC" | cut -f1)
DST_SIZE=$(du -sb "$DST" | cut -f1)

echo "SRC: $SRC_COUNT files, $SRC_SIZE bytes"
echo "DST: $DST_COUNT files, $DST_SIZE bytes"

if [ "$SRC_COUNT" = "$DST_COUNT" ] && [ "$SRC_SIZE" = "$DST_SIZE" ]; then
    echo "=== $(date) VERIFIED MATCH -- removing source $SRC ==="
    rm -rf "$SRC"
    echo "=== $(date) DONE. output_vla fully moved to $DST ==="
else
    echo "=== $(date) MISMATCH -- NOT removing source. Manual check needed. ==="
    exit 1
fi
