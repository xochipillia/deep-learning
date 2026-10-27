# Data and Credential Safety

Before publishing changes:

1. Keep datasets, model weights, caches, raw records, and generated examples
   out of Git.
2. Clear all notebook outputs.
3. Do not commit `.env`, AWS configuration, access tokens, bucket names,
   account identifiers, email addresses, or absolute home-directory paths.
4. Run the assignment hygiene scanner:

   ```bash
   bash assignments/imageclef-test-time-scaling/scripts/scan_package.sh \
     assignments/imageclef-test-time-scaling
   ```

5. Inspect staged files with `git diff --cached` before pushing.
