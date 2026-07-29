# Quran Studio AI Alignment Engine

Python API foundation for Quran Studio AI.

## Current capabilities

- Secure Bearer API authentication
- Health endpoint
- Media-format validation
- Multipart upload validation
- Detection-settings validation
- Alignment-job creation
- Job-status retrieval
- Job cancellation

## Not implemented yet

- FFmpeg audio extraction
- Arabic speech recognition
- Full verified Quran corpus
- Surah and Ayah matching
- Word-level alignment
- Persistent jobs
- Caption-result generation

The service deliberately returns `not_ready` until the model and verified Quran
corpus are connected.