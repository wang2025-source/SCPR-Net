# Model Zoo

All listed files are stored under `code/model/`. Filenames are intentionally unchanged so the original code and experiment records remain traceable.

| Checkpoint | Approx. size | Purpose |
|---|---:|---|
| `EMMA_SOURCE_WAVE_ASSM_V10_SCREEN_best_pareto.ckpt` | 38.1 MiB | Source/wavelet/ASSM V10 stage |
| `EMMA_V10_REGION_DETAIL_V12_FIXED_STRONG.ckpt` | 12.8 MiB | Region-detail V12 stage |
| `EMMA_REGION_MOE_V22.ckpt` | 39.4 MiB | **Selected best/final SCPR-Net model** |

The portable SCPR inference chain is:

```text
V10 configuration → V12 architecture/state → V19/V22 region-MoE state
```

Ai/Av, the EMMA baseline, V19 and other experimental checkpoints are not committed because they are not used by the selected V22 inference chain.

Use `python tools/verify_snapshot.py` to verify the exact hashes of the included checkpoints.
