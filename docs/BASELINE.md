# Phase 0 baseline (original system, LLM disabled)

Reproduce: `python -m evaluation.run --suite all --mode full --out reports/baseline.json`
(reports for each suite are in `reports/baseline_*.json`). Live search is used, so claims and
scam-full numbers can move slightly between runs until the Phase 1 cache lands.

| Suite | n | Result |
|---|---|---|
| Phishing URL (OpenPhish vs Tranco, URL only) | 300 | accuracy 0.50, phishing recall **0.00**, best phishing score 0.45 (< 0.6 threshold) |
| Claims (hand-labelled) | 40 | accuracy 0.40, macro-F1 0.35; supported recall 0.33, refuted recall 0.07 |
| Scam cases, fast (no claim checking) | 30 | deceptive recall 12/14; genuine 0/10 cleared (no evidence, abstains) |
| Scam cases, full (live evidence + NLI) | 30 | accuracy 0.77, macro-F1 0.57; deceptive 13/14, genuine 10/10, **ambiguous 0/6 (all called genuine)** |

## Findings that drive the plan

1. **URL heuristics miss modern phishing.** Real phishing is hosted on free platforms
   (r2.dev, pages.dev, vercel.app) with no keyword cues. -> Phase 3.
2. **Evidence text is poor.** Wikipedia snippets contain raw `<span class="searchmatch">` HTML and are
   tiny fragments; news "passages" are only headlines with inflated similarity (0.95). (An apparent
   mojibake finding was a terminal display artifact and was retracted: stored text is clean.) -> Phase 2.
3. **Unsafe reassurance.** Any retrieved evidence, even irrelevant, lets the verdict become `genuine`
   (`has_evidence=bool(evidence)` in `case_analysis.py`). -> Phase 2 (relevance-gated evidence) and
   Phase 7 (abstention rules).
4. **Image OCR uses a single-line model on whole images** (TrOCR-base-printed). -> Phase 4.
5. **LLM can override the verdict** (`case_analysis.py`). -> Phase 1.
6. **Provider flakiness.** Wikipedia returned HTTP errors mid-run and a 60 s cooldown silently drops a
   source. No cache, so runs are not reproducible. -> Phase 1.

## Caveats on these numbers

- The scam set is small (30) and hand-written; the deceptive cases contain keywords by construction, so
  deceptive recall is optimistic. It grows in later phases.
- The claims set is small (40); treat differences under ~0.1 as noise.

## Additional findings after Phase 1 (from diagnosing one flipped case)

7. **Sentence splitter broke on decimals and abbreviations** ("Python 3.13" became two claims). Fixed in Phase 2.
8. **Unrelated passages could refute a claim.** Non-life-status claims skipped the relevance check entirely, and
   the verifier's "explicit negation" rule accepted any "not" in any passage. A benign message was called
   `deceptive` because an unrelated "HCL Notes" snippet contradicted "13 release notes are available".
