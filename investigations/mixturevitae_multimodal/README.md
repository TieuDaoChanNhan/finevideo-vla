# mixturevitae_multimodal/

Scratch scripts from a side investigation into an **external** HF dataset —
`mixture-vitae-backup/MixtureVitae-Backup/data/multimodal` (SNAC audio tokens + VALID text/image
captions) — to evaluate whether it's worth mixing into the VLA training corpus.

**This is not core pipeline code.** It's unrelated to the project's own multimodal token pipeline
(`pipeline_video/`, `pipeline_pose/`). Findings from this investigation are written up in
`../../PROGRESS.md` under "MixtureVitae-Backup Multimodal Investigation" — read that first.

**Status:** paused, awaiting a go/no-go decision from the team before any further work here.

| File | Purpose |
|------|---------|
| `collapse_snac.py` | Merges audio SNAC tokens + image caption + emotion label into `<speak>`/`<listen>` tagged sequences. Hardcoded to read from `/mnt/sda/snac1..5/` (a specific cluster mount). |
| `insert_valid.py` | Processes "VALID" dataset text/image records — language filtering, CJK detection, image handling. **Does not run as-is** — imports `fem.json`, `names`, `flagged_words` which are not present anywhere in this repo (likely copy-pasted from another codebase during exploration). Kept for reference logic only. |
| `head.txt` | Sample records from `ontocord/VALID` (head only, not the full dataset). |

If you pick this investigation back up, the newer (and working) approach is
`../../tools/inventory/peek_multimodal.py` + `../../tools/inventory/count_multimodal_tokens.py` —
true HTTP streaming, no full download, resumable checkpoint. Prefer extending those over fixing the
scripts in this folder.
