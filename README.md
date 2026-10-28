# SignalCheck

SignalCheck is a research prototype for **multimodal, evidence-based content-risk assessment**. Give it a text
message, a link, a screenshot or photo, and a voice note, alone or together as one case. It checks each input,
compares them with each other, and answers `genuine`, `deceptive` or `needs_review`, with the reasons and the
evidence behind them.

It does **not** prove that anyone lied. It assesses the content and the risk of trusting it; sender intent is always
reported as not determinable. `genuine` means "no risk found and something positive supports it", not "authentic".

## Design principles

1. **Abstain rather than guess.** Without supporting evidence the answer is `needs_review`, never `genuine`.
2. **Warnings are cheap, reassurance is expensive.** A verdict needs corroboration (two independent sources, or one
   official/reputable one), support needs stronger model confidence than refutation, and images, audio and cross-channel
   checks can only *raise* risk. Only a supported claim or a trusted, brand-consistent link can clear a case.
3. **The language model is an assistant, not the judge.** The optional Groq investigator can only turn `genuine` into
   `needs_review`. It never clears anything and never makes a case `deceptive` on its own.
4. **Everything runs on pretrained models and free, keyless sources** (CPU only). No training.

## What it does

| Channel | What is checked |
|---|---|
| **Text** | Factual claims are extracted (personal messages, instructions, coordination and time-relative local statements are skipped), searched on Wikipedia, DuckDuckGo and Google News, filtered for relevance, and judged by a DeBERTa NLI model. Quoted or debunking passages cannot support a claim; fact-checkers can refute but never support. Scam language is scored separately. |
| **Link** | Lookalike and typosquat domains, brand names in subdomains or paths, free-hosting abuse, random labels, punycode, short-link expansion (with SSRF guards), domain age (RDAP), public phishing feeds, and a popularity ranking used to earn trust. |
| **Image** | Full-page OCR (RapidOCR), scam language inside screenshots, addresses shown in them, brand shown versus domain linked, EXIF editing software, and CLIP image-caption consistency for photos. |
| **Voice** | Local speech-to-text (faster-whisper) with language detection and translation to English, then the same text checks plus spoken-scam patterns (authority impersonation, gift-card demands, remote access, one-time-code read-outs, family-emergency). |
| **Across channels** | Different senders named in different inputs, conflicting contact details or amounts, pressure repeated in several channels, and NLI contradictions between claims made in different inputs. |

Each result carries a deterministic explanation (`explanation`, every line linked to the inputs or evidence behind it)
and a per-signal `risk_breakdown`.

## Results (development benchmarks, see `docs/BASELINE.md` and `docs/PROGRESS.md`)

| Measure | Original system | Now |
|---|---|---|
| Claims (40 labelled): accuracy, with precision 1.00 on supported and refuted | 0.40 | 0.875 |
| Real phishing URLs called `genuine` (URL signals only, feeds off) | 139 / 150 | 0 / 300 across two samples |
| OCR word recall on screenshots | 0.00 (TrOCR) | 0.998 |
| Speech word error rate, real human speech | n/a | 6.1% |
| Multi-channel scam benchmark (24 cases): scams caught / genuine cleared / wrong confident answers | n/a | 12/12 / 6/6 / 0 |

**Read these with care.** The benchmarks are small and several were built by the author, so they are development
numbers, not unbiased estimates. Image and audio tests use rendered screenshots and text-to-speech, so they check the
logic more than real-world robustness. The ablation shows that on the multi-channel benchmark only the
cross-channel step turned two individually-weak signals into a catch; the screenshot and voice note added nothing
alone. Fresh, unseen inputs are the honest test.

## What it cannot do

- Prove intent, or tell whether a voice or an image is synthetic or edited (no deepfake or AI-image detection).
- Verify a text-free photo, or verify personal and local statements (they are left as `needs_review`).
- Guarantee that free search results are complete. Wikipedia rate-limits generic clients; set `WIKIMEDIA_CONTACT`.
- Read non-English screenshots (the OCR handles English and Chinese). Spoken languages are translated by Whisper.
- Persist cases: they live in process memory and vanish when the backend stops.
- Provide calibrated probabilities: risk scores are signal strengths.

## Setup (Windows)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
Set-Location frontend; npm install; Set-Location ..
```

Create a local `.env` (never commit it):

```text
GROQ_API_KEY=your-key          # optional: enables the escalate-only AI investigator
WIKIMEDIA_CONTACT=you@example.org   # optional: polite contact for Wikipedia
```

Other settings: `NLI_MODEL`, `WHISPER_MODEL` (default `small`), `GROQ_MODEL`, `GROQ_MIN_INTERVAL_SECONDS`,
`EVIDENCE_CACHE=off`, `URL_NETWORK_CHECKS=off`, `REPUTATION_DATA=off`, `VISION_MODELS=off`.
Models download on first use (DeBERTa, CLIP and Whisper `small` are a few hundred MB each).

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Set-Location frontend; npm run dev
```

Open <http://localhost:5173>. API docs are at <http://127.0.0.1:8000/docs>.

## Main API routes

- `POST /api/analyze/text`, `/url`, `/image` — single-input analysis.
- `POST /api/cases`, then `POST /api/cases/{id}/inputs/{text|url|image|audio}`, then `POST /api/cases/{id}/analyze`.
  Audio is transcribed in the background; `analyze` answers 409 until it finishes.
- `GET /api/cases/{id}`, `DELETE /api/cases/{id}`, `GET /health`.

## Test and evaluate

```powershell
.\.venv\Scripts\python.exe -m pytest -q                       # offline and hermetic (253 tests)
.\.venv\Scripts\python.exe -m evaluation.run --suite claims   # phishing_url | claims | scam | image | audio | ablation | all
```

Reports are written under `reports/`. Documents: `docs/REWORK_PLAN.md` (plan), `docs/BASELINE.md` (starting point),
`docs/PROGRESS.md` (every phase with results and caveats).

This is the updated progrss
