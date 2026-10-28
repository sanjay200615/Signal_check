# Rework plan

Goal: turn SignalCheck from a text pipeline with OCR in front into a genuinely multimodal, evidence-grounded content-risk system that follows the target architecture (modality adapters -> shared claim representation -> evidence retrieval/verification + cross-modal fusion -> cross-channel correlation -> decision fusion -> explanation).

## Constraints

- No model training. Pretrained models and free services only.
- CPU only.
- Prefer sources that need **no API key**. A key is requested only when the keyless option is clearly worse, and each such request is raised with the user first.
- Scope: text, URL, image, audio. **No video.**
- Deadline is shorter than 6 weeks. Phases are ordered so the project is demonstrable after each one, and the cut line is marked below.
- Each phase ends with: backend tests, a user-level check (running app / real inputs), a recorded evaluation number where applicable, and a short report to the user before the next phase starts.

## Principles

1. The LLM is a helper (extraction, explanation), never the judge. Verdicts come from deterministic fusion.
2. Modalities are compared with each other, not flattened to text and forgotten.
3. Abstain (`needs_review`) rather than guess. Report the abstain rate.
4. Every claim about the system is backed by a number from the evaluation harness.
5. Every network call is cached (SQLite) so runs are reproducible and rate limits are survivable.

## Tool choices (keyless first)

| Need | Primary (no key) | Optional upgrade (key) |
|---|---|---|
| Web evidence | Wikipedia, Wikidata, Google News RSS, GDELT DOC API, `ddgs` (DuckDuckGo) | Tavily |
| Existing fact-checks | ClaimReview markup scraped from search results, Wikipedia | Google Fact Check Tools API (free key) |
| Phishing feeds | OpenPhish community feed, URLhaus/PhishTank bulk dumps where keyless | Google Safe Browsing, VirusTotal |
| URL technical checks | `tldextract`, DNS/RDAP domain age, TLS cert, redirect chain | |
| OCR | RapidOCR (ONNX), EasyOCR fallback | |
| Image-text, brand | CLIP ViT-B/32 or SigLIP-base (local) | |
| Image reasoning | CLIP zero-shot + OCR + local heuristics | Hosted VLM (Groq/Gemini free tier) |
| Speech-to-text | faster-whisper small/base, int8 | |
| Embeddings / rerank | bge-small, bge-reranker-base | |
| NLI | DeBERTa-v3-base MNLI/FEVER/ANLI | |
| Entities | spaCy `en_core_web_sm` | |
| LLM (extraction, explanation) | Local small instruct model via llama.cpp if fast enough, else template explanations | Groq (already configured) |

## Phases

### Phase 0 - Baseline and evaluation harness
- Read and document every unread service; record the current behaviour.
- Evaluation package: dataset loaders, metrics (macro-F1, abstain rate, coverage-vs-error), runner producing a JSON/markdown report.
- Small public subsets: claim verification, phishing URLs, SMS/email phishing, image-caption mismatch.
- Cross-channel scam benchmark (template-generated, hand-checked), grown over time.
- Baselines: current system, LLM-only.
- Exit: one command prints baseline numbers for the current system.

### Phase 1 - Core refactor (the only heavy restructuring)
- Modality adapter interface; shared `Claim` and `Signal` objects; provider interface; SQLite cache; job runner.
- Port existing routes and tests onto the new interfaces; all existing tests stay green.
- Remove the LLM verdict override.
- Exit: same behaviour as before, new structure, tests green, numbers unchanged within noise.

### Phase 2 - Text and evidence quality
- DeBERTa NLI, bge reranker, entity extraction, keyless providers (Wikidata, GDELT, ddgs), better query generation.
- Exit: evaluation shows improvement over the Phase 0 baseline.

### Phase 3 - URL depth
- Domain age, TLS, redirect chain, feed lookups, lookalike/typosquat detection.
- Exit: phishing URL subset metrics recorded.

### Phase 4 - Image
- Full-page OCR replacement, CLIP image-text consistency, brand/logo vs domain, EXIF, weak AI-image signal (clearly labelled).
- Exit: image-caption mismatch and screenshot cases behave as intended.

### Phase 5 - Audio
- faster-whisper transcription; transcript flows through the text pipeline and scam-cue checks.
- Exit: a real voice note produces a transcript and a case analysis end to end.

### Phase 6 - Cross-modal and cross-channel reasoning (the multimodal core)
- Claim clustering, NLI between claims from different inputs, brand vs domain vs claimed-sender consistency, shared identifiers.
- Exit: scam benchmark shows the cross-channel signals changing outcomes.

### Phase 7 - Fusion, explanation, UI, write-up
- Log-odds fusion with abstention; thresholds tuned on a dev split (not trained).
- Explanation narrates structured evidence only.
- Frontend shows claims, evidence links, per-modality signals.
- Ablations: text-only, +URL, +image, +audio, +cross-channel, against baselines.

## Cut line if time runs short

Keep: Phases 0, 1, 2, 3, 4, 6, 7. Drop or shrink first: Phase 5 (audio), then the benchmark size, then optional upgrades.

## Known limits (stated up front)

- Deepfake / AI-image detection with small pretrained models is unreliable; reported only as a weak labelled signal.
- Sender intent is not determinable from content.
- Free search coverage is the ceiling for fact-checking; uncovered claims abstain.
- Risk scores are not calibrated probabilities.
