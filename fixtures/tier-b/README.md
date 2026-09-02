# Tier B live benchmark scenes (N=10)

Photos are **not** committed by default. After clone:

```bash
chmod +x scripts/sync-lab-images.sh
./scripts/sync-lab-images.sh
```

Catalog SSOT: `configs/fixtures.yaml` (`tier_b`). Each scene syncs to `fixtures/<id>/photo.jpg` from
`teselio-engine-py/lab/storage/assets/inputs/<lab_image>`.

Record fal replay baselines (optional, for CI without FAL_KEY):

```bash
FAL_KEY=... ./scripts/record-fal-fixture.sh terminados-04
```
