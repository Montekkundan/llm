# Benchmark Tracking

When comparing reruns, keep one table per release with:

- repo commit
- base checkpoint tag and step
- SFT checkpoint tag and step
- CORE metric
- ChatCORE metric
- branding smoke result
- identity dataset checksum

Sources:

- `run_manifest.json`
- `report/`

This keeps base, SFT, and later branded reruns comparable without relying on memory.
