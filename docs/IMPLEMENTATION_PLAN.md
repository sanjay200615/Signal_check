# SignalCheck Implementation Plan

## Project direction

SignalCheck is a research prototype for manually submitted multimodal cases.
Its focused objective is to verify claims and identify content-level
deception risk in suspicious messages, screenshots, and URLs.

It does not prove that a sender intentionally lied. The system reports whether
content is supported, refuted, high-risk, or insufficiently evidenced, while
sender intent remains `not_determinable_from_content`.

## Current decisions

- **Project type:** Research prototype
- **Current workflow:** Manual case submission
- **Primary modalities:** Text, image/screenshot, and URL
- **Audio:** Transcript-backed asynchronous ingestion exists; hosted speech
  transcription is deferred
- **Processing:** Live evidence and optional hosted AI are replaceable; a
  controlled local evidence path remains available
- **Storage:** Raw uploads are temporary process-local bytes and are not
  returned by the API
- **Primary use case:** Phishing and suspicious-content verification supported
  by claim evidence
- **Result model:** Content status, deception risk, abstention, evidence
  provenance, recommendation, and explicit limitations
- **Evaluation:** Controlled prototype dataset first; no general accuracy claim
  from hand-picked examples
- **Deployment direction:** Manual analysis now; continuous monitoring is
  deferred

## Completed implementation

- Temporary case creation, retrieval, deletion, and related input ingestion.
- Text, URL, image, and transcript-backed audio inputs.
- Live Wikipedia and Google News retrieval.
- Optional Tavily retrieval.
- Local FAISS retrieval path for reproducible demonstrations.
- Claim extraction, source-linked evidence, relevance filtering, and BART NLI.
- Source quality, provider, domain, publication, and retrieval provenance.
- URL and message-context phishing signals.
- Screenshot OCR through TrOCR.
- Combined text, URL, and image case analysis.
- Cross-input shared URL, entity, and similarity correlation.
- Conservative `needs_review` behavior when evidence is absent or weak.
- Contradiction filtering to avoid treating generic contextual headlines as
  proof that a claim is false.
- Separate content status, deception risk, and sender-intent fields.
- Optional Groq contextual investigation with deterministic fallback.
- Frontend manual-analysis workflow.
- Backend regression tests and frontend production build.

## Phase gates

Every phase must:

1. Implement only its defined scope.
2. Add or update focused tests.
3. Run backend tests and the frontend build where applicable.
4. Manually exercise the changed behavior.
5. Record verified behavior and limitations in
   [docs/PROGRESS.md](PROGRESS.md).

## Phase A — Reliable claim verification

### Goal

Reduce false refutations and make evidence aggregation defensible.

### Work

- Add claim-type-aware checks for life-status, numerical, date, location,
  identity, and causal claims.
- Prefer passages that explicitly state a claim or its negation.
- Separate article headlines from article body text.
- Record why evidence was accepted or rejected.
- Detect duplicate or copied sources so repetition is not counted as
  independent confirmation.
- Add freshness handling for current-event claims.

### Exit criteria

- Controlled tests cover supported, refuted, conflicting, and unknown claims.
- Generic contextual headlines do not create a refuted result.
- Every final status has inspectable evidence and a reason.

## Phase B — Phishing investigation

### Goal

Make suspicious-message and suspicious-URL analysis the strongest useful
demonstration.

### Work

- Inspect redirects and final landing domains where safe.
- Extract page title, forms, password fields, PIN fields, payment fields, and
  brand references.
- Detect sender/domain mismatch when sender metadata is provided.
- Add domain and certificate signals without treating them as proof.
- Compare claimed brands with configurable trusted domains.
- Display each risk indicator with its source and confidence.
- Return an actionable recommendation.

### Exit criteria

- Controlled legitimate and phishing cases are both represented.
- The result explains each major phishing signal.
- Inaccessible pages produce `needs_review`, not `genuine`.

## Phase C — Genuine image analysis

### Goal

Move beyond OCR while preserving uncertainty.

### Work

- Check whether the configured hosted account exposes a vision-capable model.
- Add an adapter that sends image pixels only when supported.
- Return structured scene observations, visible text, claim-relevant objects,
  manipulation indicators, confidence, and limitations.
- Keep “unusual” separate from “manipulated.”
- Compare image observations with accompanying text and URL.
- Never infer authenticity from missing EXIF data alone.

### Exit criteria

- Text screenshots use OCR and claim verification.
- Non-text images return visual observations or an explicit unavailable status.
- Unusual images can produce `needs_review` rather than an unsupported
  genuine/deceptive claim.

## Phase D — Coordination analysis

### Goal

Make cross-input coordination more than simple similarity.

### Work

- Track repeated claims and URLs across inputs.
- Compare image perceptual hashes.
- Detect near-duplicate wording and translated variants.
- Record timestamps where available.
- Distinguish shared origin from independent corroboration.
- Present a structured coordination explanation.

### Exit criteria

- Cases show why inputs appear coordinated.
- Shared content is not incorrectly counted as independent evidence.

## Phase E — Controlled evaluation

### Goal

Measure the prototype instead of relying on selected demonstrations.

### Dataset groups

- Supported, refuted, conflicting, outdated, and unknown claims
- Legitimate messages
- Bank, delivery, job, government, and account-recovery scams
- Genuine and edited screenshots
- Text-bearing and non-text images
- Consistent and inconsistent multimodal cases

### Metrics

- Claim-status accuracy
- Phishing precision, recall, and F1
- False-positive and false-negative rates
- Abstention rate
- Evidence-source coverage
- Confidence calibration
- Coordination precision

### Exit criteria

- Cases have independent labels and review notes.
- Results document limitations and uncertainty.
- No general accuracy claim is made from demo cases alone.

## Phase F — Audio and video

Audio and video remain secondary until the text, phishing, image, and
evaluation workflow is reliable. Add them only with real transcription or
feature extraction, temporary media processing, asynchronous status tracking,
modality-specific evidence, and evaluation data.

## Non-goals

- Proving that a sender intentionally lied.
- Automatically declaring every unusual image fake.
- Presenting heuristic risk scores as probabilities.
- Claiming complete internet coverage.
- Building permanent surveillance or permanent upload storage.
