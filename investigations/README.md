# investigations/

Scripts for probing or converting **external** datasets — evaluating whether they're worth pulling
into the FineVideo-VLA training corpus, or making an external dataset's tokens vocab-compatible.
Separate from the project's own pipeline (`pipeline_video/`, `pipeline_pose/`), which only ever
touches FineVideo.

| Folder | External source | Status |
|--------|-----------------|--------|
| `mixturevitae_multimodal/` | `mixture-vitae-backup/MixtureVitae-Backup/data/multimodal` (HF) | Paused — see its own README |
| `mv_omni_seed_conversion/` | MixtureVitae-Omni `<seed_N>` → this project's `<seed2_N>` vocab | Done, results in `../PROGRESS.md` |
