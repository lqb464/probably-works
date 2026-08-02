# Diagnostic Scope

This folder contains the standalone global-hidden recovery diagnostic for ITSELF.

Rules for future edits:

- Keep diagnostic code isolated in this folder.
- Do not modify training code, losses, optimizers, schedulers, datasets, or checkpoint contents.
- Treat the loaded checkpoint as frozen evaluation-only state.
- Keep the primary hidden scorer parameter-free.
- Preserve the official global evaluation path as the fidelity reference.

