# Implementation progress

## Status

Current phase: **Core manual verification milestone complete; Phase A — reliable claim verification next**

Started: 2026-10-27

## Phase 1

### Scope

Create a temporary, manual case container that can hold related text, URL, and image inputs. This phase establishes ingestion and lifecycle behavior only; it does not produce a fused verdict.

### Completed

- Phase scope and exit criteria documented.
- Implementation task dependencies recorded.
- Added `POST /api/cases` for creating a temporary manual case.
- Added text, URL, and image input routes under `/api/cases/{case_id}/inputs`.
- Added stable input IDs, SHA-256 content hashes, timestamps, and input metadata.
- Kept uploaded image bytes in process memory only and excluded them from responses.
- Added case retrieval and deletion routes.
- Added focused case API tests.
- Provider initialization now loads the local embedding model and FAISS index only for `EVIDENCE_PROVIDER=local`.
- The default live provider now combines Wikipedia reference search with Google News.
- Wikipedia and encyclopedia evidence receive reputable-source weighting. Tavily remains optional and is included automatically when `TAVILY_API_KEY` is configured.
- Added an optional Groq AI investigator for combined cases. When `GROQ_API_KEY` is configured, it inspects supplied message text, safely fetched public-page text, retrieved claims, and evidence to produce a contextual verdict, risk score, confidence, summary, and reasons. The deterministic fusion result remains the fallback when the key is absent.
- Groq model selection uses `GROQ_MODEL`; the default is `openai/gpt-oss-20b`, selected from the models available to the configured account during local verification.
- Live provider startup no longer requires local index artifacts.
- Health readiness now reports `analysis_ready` separately from optional local retrieval components.
- Restored frontend dependencies and verified the Vite production build.
- Generated the local FAISS index from the controlled corpus: 8 passages indexed.
- Validated the real Sentence Transformer retrieval model and BART NLI model.
- Added modality-neutral contracts for text, image, URL, audio, and video.
- Added a reusable factual-claim service and wired the main text pipeline to use it.
- Added deterministic URL phishing-risk signals that do not require external APIs.
- Added lightweight image dimensions, format, metadata, and perceptual-hash assessment.
- Combined case analysis now runs OCR on uploaded images, verifies OCR-derived claims through the normal evidence pipeline, and exposes the extracted text in image signals.
- Image-only inputs without readable text now return `needs_review` instead of being silently classified as genuine. This is intentionally conservative because no pixel-level vision/manipulation detector is configured yet.
- Combined decisions now abstain as `needs_review` when no relevant source-linked evidence is available. The response includes a decision basis, recommended action, limitations, and evidence provenance fields.
- The UI labels the risk value as a prototype score and displays source quality/provider information to avoid presenting heuristic scores as calibrated probabilities.
- Combined results now separate content status (`supported`, `refuted`, or `insufficient_evidence`) from deception risk and explicitly report that sender intent is not determinable from content alone.
- Added cross-input shared URL/entity/similarity correlation.
- Added documented weighted decision fusion for the first genuine/deceptive demonstration.
- Added asynchronous audio case ingestion with temporary raw bytes and transcript adapter support.
- Added `POST /api/cases/{case_id}/analyze` to combine available case signals into one typed result.
- Added backend integration coverage for claim contradiction, phishing risk, and cross-input correlation.
- Added a frontend “Analyze combined case” action using the collected text, URL, and image inputs.

### Verification

- Focused Phase 1 and health tests: **4 passed**.
- Full backend regression suite: **49 passed**.
- Registered routes inspected for case creation, retrieval, deletion, and all three input types.
- Full backend regression suite after stabilization changes: **52 passed**.
- Python syntax compilation: **passed**.
- Frontend `npm run build`: **passed**.
- Live-provider startup isolation test: **passed** without a local index.
- Local index generation: **passed**; `data/index/evidence.faiss` and `data/index/metadata.json` were created.
- Local provider preflight after index generation: **passed**.
- Real model-loading preflight for retrieval and NLI: **passed**.
- Full backend regression suite after index generation: **54 passed**.
- Full backend regression suite after Phases 3–9 implementation: **64 passed**.
- Audio case API test: **passed** with asynchronous `202` response and temporary input metadata.
- End-to-end case integration test: **passed** with a deceptive fused result.
- Full backend suite after case integration: **65 passed**.
- Frontend production build after case integration: **passed**.
- Full backend regression suite after abstention, provenance, intent-separation, and contradiction-guard changes: **78 passed**.
- Frontend production build after result-model and UI changes: **passed**.
- Live backend health check after restart: **ready**.
- Live text smoke test completed successfully; an ambiguous Earth-shape headline no longer creates a direct refutation.
- Live combined phishing smoke test completed successfully with `deceptive`, high deception risk, and `not_determinable_from_content` intent.

### Remaining limitations

- Case storage is expected to be process-local during the research prototype.
- Advanced hosted transcription is still deferred; audio cases currently require a transcript for completed processing.
- Raw uploads must remain temporary and must not be exposed in API responses.
- The generated index is ignored by Git and must be rebuilt by each environment that uses local retrieval.
- The current audio implementation accepts a supplied transcript through a replaceable adapter; a real hosted speech-to-text provider still needs configuration.
- The current case endpoint includes image metadata and OCR signals, but it does not yet infer image manipulation risk from pixels.
- URL analysis uses deterministic local signals; external reputation providers are not required yet.
- Sender intent is not inferable from content alone; `deceptive` means high content-risk, not proven malicious intent.
- The prototype risk score is not a calibrated probability.

## Phase history

| Phase | Status | Notes |
|---|---|---|
| 1. Manual case workflow | Complete | Case creation, related input ingestion, metadata, temporary raw content, retrieval, deletion, and tests are in place |
| 2. Prototype stabilization | Complete | Provider separation, preflight, index generation, real model loading, frontend build, and regression tests passed |
| 3. Common modality schema | Complete | Contracts cover planned modalities and validate scores/statuses |
| 4. Claim-verification service | Complete | Reusable factual-claim extraction boundary is wired into the text pipeline |
| 5. Phishing detection | Complete | Deterministic URL/page risk assessment implemented and tested |
| 6. Image analysis | In progress | Metadata, OCR, OCR claim verification, and conservative `needs_review` behavior implemented; pixel-level vision analysis remains |
| 7. Cross-input correlation | Complete | Shared URL/entity/similarity signals implemented and tested |
| 8. Decision fusion | Complete | Weighted risk fusion with abstention, decision basis, recommendation, and intent separation implemented and tested |
| 9. Audio support | Complete with limitation | Async audio case ingestion and transcript adapter implemented; hosted STT remains |
| Integration milestone | Complete with limitation | Backend and frontend run combined manual cases; advanced image/audio models, calibrated evaluation, and deeper phishing inspection remain future work |

## Rework (see REWORK_PLAN.md and BASELINE.md)

### Phase 0 - baseline and evaluation harness: complete
- `evaluation/` package (metrics, loaders, runner), 40 labelled claims, 30 scam cases, live OpenPhish/Tranco URL set.
- Baseline recorded in `docs/BASELINE.md`.

### Phase 1 - core refactor: complete
- Modality adapters (`app/services/adapters.py`); `analyze_case` is now orchestration only.
- SQLite evidence cache with retry/backoff (`app/services/cache.py`), applied per source so failures are never cached.
- LLM investigator is advisory only and can no longer change the verdict.
- Verified: 98 tests pass; deterministic scam results identical to baseline; warm-cache reruns identical and 4x faster (418s -> 157s -> 100s); real server flow and error codes checked; frontend builds.

### Phase 2 - text and evidence quality: complete
Changes (each has regression tests; 118 pass):
- Sentence splitter no longer breaks decimals/abbreviations ("Python 3.13", "$2.99", "Dr.").
- Check-worthiness filter: personal/transactional statements ("your order has shipped"), instructions,
  conditionals and interjections are no longer fact-checked against the web (this removed false
  "refutations" of genuine notifications). Trade-off: generic second-person advice ("wash your hands") is skipped.
- Relevance gate: evidence must cover >= half of the claim's content terms.
- Wikipedia HTML stripped; AI-generated mirrors (grokipedia) down-weighted; fiction/entertainment
  articles (title suffix "(film)", "(novel)"...) excluded as evidence.
- New keyless source: DuckDuckGo (`ddgs`) with real body snippets.
- NLI default is now DeBERTa-v3 (A/B on the claims suite: 0.75 -> 0.85 accuracy vs BART); configurable via `NLI_MODEL`.
- Verifier: debunking passages ("conspiracy", "myth", "hoax"...) cannot count as support; a confident
  contradiction on a closely matching passage now refutes.
- Fusion: risk in [0.3, 0.6) is `needs_review`, never `genuine`.

Results (live search, LLM disabled; see BASELINE.md for the starting point):

| Suite | Baseline | After Phase 2 |
|---|---|---|
| Claims (n=40) accuracy / macro-F1 | 0.40 / 0.35 | 0.90 / 0.89 |
| Claims refuted recall | 0.07 | 0.80 |
| Scam (n=30) accuracy on answered | 0.77 (all answered) | 1.00 (abstain 0.47) |
| Scam deceptive recall / precision | 13/14 / 1.00 | 12/14 / 1.00 |
| Scam genuine cleared | 10/10 (by luck; 6/6 ambiguous also "genuine") | 4/10 (rest abstain); ambiguous 6/6 held |

Decisions and caveats:
- Reranker and spaCy entities were in the plan but are **not added**: accuracy is already 0.90 and the lexical
  coverage gate does the job; revisit only if a measured failure needs them.
- The system now abstains instead of guessing. Genuine recall is low because free web evidence cannot
  confirm personal notifications; Phase 3 (URL/domain reputation) is how `genuine` gets established.
- Known remaining errors: a local-claim ("library will extend hours") is "supported" by a generic page;
  3 refuted claims stay `insufficient_evidence`.
- The scam set is small and hand-written, and some heuristics (imperative verb list) were tuned after seeing
  it, so treat its numbers as development-set results, not unbiased estimates.

### Phase 3 - URL and message depth: complete
Changes (154 tests pass; tests are hermetic via `tests/conftest.py`):
- `reputation.py`: Tranco top-100k, OpenPhish feed, Phishing.Database host list, brand -> official-domain map, PSL-based
  multi-tenant platform detection. All keyless, cached under `data/reputation/`, degrade to "no information" offline.
- `domain_analysis.py`: homoglyph / typosquat / same-label-other-TLD lookalikes, brand in subdomain / domain / path,
  free-hosting platforms, random machine-generated labels, punycode.
- `url_network.py`: domain age via RDAP (IANA bootstrap, cached) and short-link expansion with SSRF guards
  (public addresses only, standard ports, every hop re-validated).
- `phishing.py`: structured message cues (credential request, threat/urgency, financial bait, advance fee, crypto
  doubling) that require request/threat phrasing; works with no URL. Message-names-brand-but-links-elsewhere signal.
- Fusion: `genuine` can now be earned by a trusted domain (popular, no indicators, brand-consistent, not a shortener
  or multi-tenant host); risk always overrides trust. Phishing weight is 1.0 so the 0.6 line means the same thing
  in `assess_url` and in fusion. `needs_review` with risk >= 0.3 now explains which indicators fired ("elevated").

Results (see reports/phase3*.json):

| Suite | Before | After Phase 3 |
|---|---|---|
| URL-only, dev sample (OpenPhish vs Tranco), feeds OFF: phishing flagged deceptive | 0 / 150 | 23 / 150 |
| ... phishing called genuine | 139 / 150 | 0 / 150 |
| URL-only, held-out (Phishing.Database), feeds OFF: flagged deceptive / sent to review | not measured | 13 / 137 (of 150) |
| Benign (Tranco rank 20k-100k) falsely deceptive | not measured | 0 / 150 |
| Scam cases (n=30): accuracy / macro-F1 | 0.73 / 0.70 | 0.93 / 0.92 (fast mode; full mode was 0.90 / 0.87 before the shortener fix) |
| Scam genuine cleared | 4 / 10 | 10 / 10 |

Findings and caveats:
- URL-only detection is capped: modern phishing sits on free hosting with few URL tells. Most of it goes to
  `needs_review` rather than `deceptive`. Catching more needs page analysis (Phase 4/6 brand-vs-domain checks).
- Free feeds barely overlap: 0 / 150 independent Phishing.Database URLs appear in the OpenPhish feed. The
  in-sample 150 / 150 only proves the lookup works and is not a detection rate.
- Bugs found by evaluation and fixed: feed lookup was skipped for hosts tldextract cannot parse; user-content
  platforms (square.site, gitbook.io) inherited the platform's trust; `bit.ly` was "trusted" because it is popular;
  tests were quietly calling Groq with the real key (now blocked in conftest).
- The scam set is small and hand-written, and the brand map and cue patterns were written with it in view, so
  its numbers are development-set results. The URL held-out figure (13 / 150) is the more honest one.
- Known remaining: `a06` (reserved domain example.org, no cues) is `genuine` while its label says `needs_review`:
  `genuine` here means "no deception risk found + reputable link", not "authentic". `s13` (.click bill scam)
  lands in `needs_review` at risk 0.55.

### Phase 4 - image: complete
Changes (172 tests pass):
- OCR: TrOCR (single-text-line model, fed whole screenshots) replaced by RapidOCR (`app/services/ocr.py`), which keeps
  line positions, reads address bars, and strips screenshot chrome ("Messages", "Today 10:42") before claim extraction.
- `vision.py`: local CLIP (ViT-B/32) for image-text similarity. Used only to check a text-free photo against a
  descriptive caption (6-60 words); mismatch below 0.237 raises `image_text_consistency` -> `needs_review`.
- `ImageAdapter`: scam language inside the screenshot, addresses shown in it (assessed like any URL), brand shown vs
  domain linked (`image_brand_domain_mismatch`), screenshot address vs link, EXIF software/camera/GPS.
- Principle: an image can only raise risk, never clear it (address bars, logos and text are forgeable). A text-free
  image is always `needs_review`.
- Brand matching tolerates OCR confusions (SBl, PayPaI) and requires capitalisation for brands that are also everyday
  words (Apple, Chase, Zoom, UPS...).

Results (`reports/phase4_image.json`; synthetic screenshots plus real Flickr30k photos):

| Test | Before | After |
|---|---|---|
| OCR word recall, 30 rendered screenshots | 0.000 (TrOCR) | 0.998 |
| Login-page mockup, lookalike address shown | not detectable | 10/10 deceptive |
| Login-page mockup, brand vs wrong link | not detectable | 10/10 deceptive |
| Login-page mockup, consistent address + link | n/a | 10/10 genuine (cleared by trusted link) |
| Scam SMS screenshots, image only | 0 (OCR unreadable) | 4/14 deceptive, 0/16 false alarms |
| CLIP caption consistency, 60 real photos | n/a | AUC 0.994, top-1 retrieval 0.92 |

Caveats and deliberate omissions:
- Screenshots and login pages are rendered by `evaluation/synth_images.py`, so those numbers show the logic works,
  not that it survives real-world screenshot noise. The CLIP result is on real photos, but its threshold was chosen on
  the same 60 pairs, so the flag rates are optimistic.
- Image-only scam detection is modest (4/14) because without a link the message wording alone is weak evidence;
  the rest go to `needs_review` rather than being cleared.
- **Not shipped, on purpose:** CLIP logo/brand detection (no labelled real-logo set to validate against, and a
  hallucinated brand would cause false alarms) and an AI-generated-image detector (without labelled AI images only
  false positives could be measured, never recall). Brand detection is OCR-based.
- Not verified in a browser: the UI already uploads images, but the new signals were only checked through the API.
- Wikimedia's REST API returns 403 to generic user-agents (robot policy). The Wikipedia evidence provider uses a
  placeholder contact address and may be affected during bursts; a real contact should be configured.

### Manual-testing fixes (found by the user testing "dua lipa dead watch video")
- Life-status relevance only understood one-word subjects, so "Is <full name> dead?" never matched any evidence.
  The subject is now the name before the life-status word (up to 3 words). A subject that is a trailing list item
  ("...and Dua Lipa dead", "...for Dua Lipa") no longer counts, because the headline is about someone else.
- Lure phrases ("watch video", "click here for the full video") are stripped from the claim text, and a new
  `clickbait_lure` message cue (sensational event + call to watch/click) lifts such messages to needs_review.
- `WIKIMEDIA_CONTACT` env var sets the contact in the Wikipedia user agent (Wikimedia rate-limits generic agents).
- Result: "dua lipa dead watch video" -> claim "dua lipa dead" refuted by the death-hoax article; combined case is
  deceptive (risk 0.95). No regression: claims 0.90 accuracy / 0.89 macro-F1, scam suite unchanged (175 tests pass).
- Repaired an accidental 4-space indent on line 1 of `app/services/cache.py` that broke all imports.

### Manual-testing fix 2: "NASA confirms 15 days of darkness" was rated "Low deception risk"
Root causes (all found by reproducing the user's screenshot case), each with regression tests (187 pass):
- A fact-check that *quotes* the hoax ("The article continues with: '[T]his time, NASA confirms...'") counted as
  support. Passages that quote the claim or call it fake news / a rumour / viral can no longer support it.
- Strong debunks (Forbes "No, NASA Is Not Predicting...", NLI 1.00) were discarded by 0.006 because the cutoff
  multiplied NLI x lexical-overlap x quality. The product no longer includes lexical overlap (kept only as a 0.3 floor).
- Conflict handling counted evidence instead of weighing it: one stray supporting passage cancelled four authoritative
  debunks ("Bats are blind"). Sides are now summed by weight; a side needs 2x the other to win, otherwise "conflicting".
- Fact-checkers (Snopes, FactCheck.org, PolitiFact, Full Fact, AFP...) are now reputable, and host matching works for
  `www.` and subdomains (the old exact match meant `www.bbc.com` never matched `bbc.com`).
- The case verdict treated *any* retained evidence as reassurance (a flaw noted in Phase 0 and wrongly deferred).
  `genuine` now needs a SUPPORTED claim (with none refuted/conflicting) or a trusted link; conflicting evidence is
  never low-risk.
- Time-relative claims with no named entity ("The city council... will close tomorrow") are not web-checkable.

Claims suite (n=40): 0.90 -> 0.95 accuracy, 0.89 -> 0.946 macro-F1; supported and refuted precision both 1.00.
Scam suite unchanged. Caveat: several of these rules were motivated by cases in this same 40-claim set, so treat
the number as development-set; manual testing with fresh inputs is the honest check.

### Escalate-only AI investigator and Groq rate-limit safety (203 tests pass)
- The Groq investigator can now only make a verdict more cautious: if the evidence-based verdict is `genuine` because
  claims were SUPPORTED, and the AI says `deceptive` with confidence >= 0.6, the verdict becomes `needs_review`
  (deception risk "elevated", explained in the decision basis). It never clears anything and never makes a case
  `deceptive` on its own.
- Live testing showed the AI is sometimes confidently wrong (it called a genuine PayPal receipt with the real paypal.com
  link "deceptive", while its own findings said the page was legitimate). So the AI is not consulted when `genuine` rests
  on a trusted, brand-consistent link, whose evidence (domain age, reputation, brand match) it cannot improve on.
- Request volume: the AI is skipped when the verdict is already `deceptive`, or `genuine` via a trusted link. Results
  are cached 24 h by content (the key ignores per-case random ids). Calls are spaced >= 3 s apart
  (`GROQ_MIN_INTERVAL_SECONDS`). A 429 starts a cooldown that honours `Retry-After` (bounded 1-300 s, default 30 s);
  during it no request is sent and the UI shows the AI as unavailable. The prompt is trimmed (2 pages x 3000 chars,
  8 evidence passages x 400 chars) to avoid token-limit 429s.
- Fixed: Groq replies with an empty `website_findings` (no page to inspect) were rejected as "invalid details".
- The investigator now refuses to fetch pages on non-public hosts (SSRF guard). NOT yet guarded: `fetch_url_text` used by
  `/api/analyze/url`, which still follows redirects without checking for internal addresses.
- Live check with the real key (6 cases): no 429, repeated case served from cache, trusted-link and deceptive cases made
  no request, 3 of 6 cases called Groq.

### Phase 5 - audio: complete (235 tests pass)
- `app/services/audio.py`: local faster-whisper (`small`, int8, CPU, no key; `WHISPER_MODEL` to change). First 10 minutes only.
  Non-English speech is detected and translated to English for analysis; the original text is kept in the metadata.
- Upload returns 202; transcription runs in the background. `analyze` returns 409 until transcription finishes. A failed
  transcription is recorded on the input and does not block the case. A supplied transcript skips speech recognition.
- `AudioAdapter`: the transcript is checked like text (claims + scam cues). A recording with no intelligible speech can
  never be cleared. Limitations state that audio analysis checks what is said, not who is speaking (cloned/synthetic
  voices are not detected).
- New spoken-scam cues: authority impersonation + threat, "don't hang up"/"don't tell anyone", untraceable payment
  (gift/prepaid cards, wire, Western Union, crypto ATM), remote-access requests, reading out a one-time code,
  unsolicited "we detected a problem" alerts, and the family-emergency scam (capped below `deceptive` on purpose, since a
  real relative could say the same words). UI: new "Voice note / call" card wired into the combined case.
- Pins: `av==16.1.0` (faster-whisper 1.2.1 breaks with PyAV 19).

Results (`reports/phase5*_audio.json`):

| Test | Result |
|---|---|
| Word error rate, 25 real human clips (LibriSpeech), small / base | 6.1% / 9.5% |
| WER, synthetic speech over a simulated phone line, small / base | 6.2% / 9.5% |
| Speed (small, CPU) | 0.34 s per second of audio |
| Spoken scam call scripts caught, clean audio (dev set: cues were tuned after a first run at 4/8) | 7/8 |
| Same scripts over a noisy phone line | 4/8 (the rest are `needs_review`, never `genuine`) |
| Benign recordings falsely `deceptive` | 0 / 24 clean, 0 / 8 phone-line |

Noise corrupts exactly the diagnostic words ("fraud" -> "fourth", "re-delivery" -> "delivery", "bitcoin" -> "bit point"),
so cue-based detection degrades faster than the overall WER suggests. Not tuned further (that would be fitting noise).

### Safety fixes found while testing audio live (also benefit text, image and URL cases)
A benign voice note ("Just confirming dinner at 7 on Friday") was rated `deceptive` (risk 0.70) because ONE irrelevant
grammar page (ludwig.guru) was judged a 0.99 contradiction. Fixes:
- Coordination statements ("just confirming", "reminder", "checking in", "running late") are not claims about the world.
- A verdict now needs corroboration: two independent sources, or one official/reputable source. A single page from an
  unknown site can neither support nor refute a claim.
- Reassurance needs stronger evidence than a warning: support needs NLI >= 0.8 (a 0.69 entailment on "state capital is
  Sydney" had made "Sydney is the capital of Australia" supported). Refutation keeps the lower bar.
- Source list widened to established references and news (Britannica, Scientific American, CNN, Bloomberg...) and
  government/university suffixes (.gov, .edu, .ac.uk, .gov.in...).
- The evidence cache stores derived fields (source quality), so entries are now versioned (`EVIDENCE_CACHE_VERSION`);
  bump it whenever that logic changes. Before this, a classification change silently had no effect for up to 7 days.

Claims suite (n=40): 0.95 -> 0.875 accuracy, 0.946 -> 0.871 macro-F1, with precision 1.00 for BOTH supported and refuted:
every error is an abstention (5 of 40), none is a wrong confident answer. This is a deliberate trade of recall for
safety; the 0.95 run had no corroboration rule and would have accused the dinner note. Scam suite unchanged (0.933).

### Phase 6 - cross-channel reasoning: complete (`app/services/cross_channel.py`)
Per-input facts (brands, phone numbers, emails, amounts, pressure cues) are compared across the inputs of a case:
different senders named in different channels, conflicting contact details or amounts, pressure repeated in two or more
channel types, and NLI contradictions between claims made in different inputs. Findings add at most 0.6 risk and can
only RAISE it: agreement between channels never clears a case. Claim clustering by embeddings was not built (lexical
overlap is used to choose which claim pairs to compare).

### Phase 7 - fusion, explanation, UI, ablation: complete (253 tests)
- Fusion stays a transparent weighted sum (not a trained model) and now reports each signal's contribution
  (`risk_breakdown`). Thresholds: >= 0.6 deceptive, >= 0.3 needs review.
- `app/services/case_explanation.py`: deterministic explanation, risks first, each line linked to input ids or evidence
  ids; no language model involved. The UI shows "Why this result" and "Where the risk score comes from".
- Ablation (`python -m evaluation.run --suite ablation`, 24 synthetic multi-channel cases, `reports/phase7_ablation.json`):

| Channels used | Scams caught as deceptive (of 12) | Genuine cleared (of 6) | Wrong confident answers |
|---|---|---|---|
| text only | 0 | 0 | 0 |
| + link | 6 | 6 | 0 |
| + screenshot | 6 | 6 | 0 |
| + voice note | 6 | 6 | 0 |
| + cross-channel reasoning | 12 | 6 | 0 |

The screenshot and the voice note add nothing on their own in this benchmark; the cross-channel step is what combines
two below-threshold signals into a catch. The scenarios were built by the author (including the one where each channel
alone falls under the threshold), so this demonstrates the mechanism, not generalisation.

- Bug found while verifying live: fact-checking sites (Snopes, FactCheck.org...) restate the claims they examine, and
  once rated reputable, such a restatement counted as support, turning the NASA hoax into "conflicting". Entailment from
  fact-checking domains no longer counts as support; their debunks still count as strong refutation.
