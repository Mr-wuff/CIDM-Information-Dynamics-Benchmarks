# Repository data and credential policy

Never commit tokens, API keys, SSH material, private data paths, training account identifiers, private user datasets, or raw diagnostic logs containing credentials. Credential-bearing notebook cells were sanitized before archive generation. The original credential may have been exposed in previous document transfers and should be revoked or rotated by its owner.

Store CDS authentication in `~/.cdsapirc` or a secured environment variable. Treat external dataset access conditions separately from code licensing. Store trained model weights and large files in release assets, institutional storage, or a dedicated data repository; do not treat GitHub Git history as a raw-data warehouse. Use Git LFS only where size and licensing permit.

No license is assigned automatically. Repository visibility should be private until the owner reviews the material and explicitly chooses the release policy.