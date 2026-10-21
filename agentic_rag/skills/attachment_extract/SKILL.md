---
name: attachment-extract
description: Validate an uploaded junior-mathematics image or PDF and extract safe file metadata or embedded PDF text before any model sees it.
---

# math.attachment_extract

Validate AttachmentUploadInput and return AttachmentExtractOutput.

## Boundaries

- Accept only JPEG, PNG, WebP, or PDF whose signature matches its declared media type.
- Enforce the configured byte, image-pixel, and PDF-page limits before model use.
- Keep file bytes in memory; never persist the attachment or place its content in Trace events.
- Extract PDF text without interpreting or solving the mathematics.
- Return safe errors for encrypted, corrupt, oversized, or mismatched files.

## Evaluation

Run contract, signature-mismatch, image-dimension, and PDF-limit cases. Security-boundary changes require adversarial upload tests.
