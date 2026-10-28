"""Run evaluation suites against the current system.

Usage:
    python -m evaluation.run --suite phishing_url
    python -m evaluation.run --suite scam --mode fast
    python -m evaluation.run --suite all --mode full --out reports/baseline.json

The LLM investigator is disabled unless --use-llm is passed, so numbers reflect
the deterministic system.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from evaluation.datasets import ROOT, load_claims, load_phishing_urls, load_scam_cases
from evaluation.metrics import classification_report

DECISION_LABELS = ["deceptive", "genuine", "needs_review"]
CLAIM_LABELS = ["supported", "refuted", "insufficient_evidence"]


def run_phishing_url() -> dict[str, object]:
    """URL-only evaluation. Feeds are disabled for the headline numbers to avoid leakage."""
    from app.services.phishing import assess_url
    from app.services.reputation import Reputation, load_reputation, normalize_feed_url
    from evaluation.datasets import CACHE_DIR, load_phishingdb_urls, sample_lowrank_benign

    full = load_reputation()
    no_feeds = Reputation(tranco=full.tranco)

    def decide(url: str, reputation: Reputation):
        """Mirror the system: deceptive on high risk, genuine only for a trusted domain, else review."""
        assessment = assess_url(url, reputation=reputation)
        if assessment.risk_score >= 0.6:
            return "deceptive", assessment
        if assessment.trusted_domain:
            return "genuine", assessment
        return "needs_review", assessment

    rows = load_phishing_urls()
    gold, predicted, trusted_phishing, trusted_benign = [], [], 0, 0
    for row in rows:
        decision, assessment = decide(row["url"], no_feeds)
        gold.append(row["label"])
        predicted.append(decision)
        if assessment.trusted_domain:
            trusted_phishing += row["label"] == "deceptive"
            trusted_benign += row["label"] == "genuine"
    report = classification_report(gold, predicted, ["deceptive", "genuine"], abstain_label="needs_review")
    report["note"] = "headline: heuristics + Tranco/brand data, phishing feeds DISABLED. genuine = trusted domain only; everything else below 0.6 risk = needs_review."
    report["trusted_domain"] = {
        "benign_marked_trusted": trusted_benign,
        "phishing_marked_trusted": trusted_phishing,
    }

    benign_low = sample_lowrank_benign(full.tranco)
    low = [decide(url, no_feeds)[0] for url in benign_low]
    report["lowrank_benign"] = {
        "n": len(low),
        "false_deceptive": low.count("deceptive"),
        "needs_review": low.count("needs_review"),
        "genuine": low.count("genuine"),
    }

    openphish_only = Reputation(
        tranco=full.tranco,
        phishing_urls=frozenset(
            normalize_feed_url(line)
            for line in (CACHE_DIR / "openphish.txt").read_text(encoding="utf-8").splitlines()
            if line.startswith(("http://", "https://"))
        ),
    )
    in_sample = [r["url"] for r in rows if r["label"] == "deceptive"]
    cross = load_phishingdb_urls()
    report["feed_coverage"] = {
        "in_sample_openphish_urls_listed": sum(openphish_only.in_feed(u) for u in in_sample),
        "in_sample_n": len(in_sample),
        "NOTE": "in-sample: the test URLs come from this very feed, so this only checks the lookup works",
        "cross_phishingdb_urls_listed_in_openphish": sum(openphish_only.in_feed(u) for u in cross),
        "cross_n": len(cross),
        "cross_heuristics_only_deceptive": sum(decide(u, no_feeds)[0] == "deceptive" for u in cross),
        "cross_heuristics_only_needs_review": sum(decide(u, no_feeds)[0] == "needs_review" for u in cross),
    }
    return report


def _url_decision(risk: float) -> str:
    if risk >= 0.6:
        return "deceptive"
    if risk >= 0.3:
        return "needs_review"
    return "genuine"


def run_claims() -> dict[str, object]:
    from app.services.pipeline import build_default_pipeline

    pipeline = build_default_pipeline(ROOT, evidence_provider="live")
    gold, predicted, details = [], [], []
    for item in load_claims():
        started = time.time()
        try:
            _, overall = pipeline.analyze(item["text"])
        except Exception as error:  # a failed lookup is an abstention, but is recorded
            overall = "insufficient_evidence"
            details.append({"id": item["id"], "error": type(error).__name__})
        if overall not in {"supported", "refuted"}:
            overall = "insufficient_evidence"
        gold.append(item["label"])
        predicted.append(overall)
        details.append(
            {"id": item["id"], "gold": item["label"], "pred": overall, "seconds": round(time.time() - started, 1)}
        )
    report = classification_report(gold, predicted, CLAIM_LABELS)
    report["details"] = details
    return report


def run_scam(mode: str) -> dict[str, object]:
    from app.services.case_analysis import analyze_case
    from app.services.case_store import (
        VerificationCase,
        build_text_input,
        build_url_input,
    )

    if mode == "full":
        from app.services.pipeline import build_default_pipeline

        pipeline = build_default_pipeline(ROOT, evidence_provider="live")
        text_analyzer = pipeline.analyze
    else:
        text_analyzer = lambda text: ([], "insufficient_evidence")  # noqa: E731

    gold, predicted, details = [], [], []
    for case in load_scam_cases():
        inputs = [build_text_input(case["message"])]
        if case.get("url"):
            inputs.append(build_url_input(case["url"]))
        verification_case = VerificationCase(case_id=case["id"], created_at="", inputs=inputs)
        try:
            result = analyze_case(verification_case, text_analyzer=text_analyzer)
            verdict, risk = getattr(result.verdict, "value", str(result.verdict)), result.risk_score
        except Exception as error:
            verdict, risk = "needs_review", 0.0
            details.append({"id": case["id"], "error": f"{type(error).__name__}: {error}"})
        gold.append(case["label"])
        predicted.append(verdict)
        details.append({"id": case["id"], "gold": case["label"], "pred": verdict, "risk": round(risk, 3)})
    report = classification_report(gold, predicted, DECISION_LABELS, abstain_label="needs_review")
    report["details"] = details
    report["mode"] = mode
    return report


def run_image() -> dict[str, object]:
    """Image pipeline evaluation on rendered screenshots (exact labels) and real Flickr30k photos."""
    import re

    import numpy as np

    from app.services.adapters import CLIP_MISMATCH_THRESHOLD
    from app.services.case_analysis import analyze_case
    from app.services.case_store import VerificationCase, build_image_input, build_url_input
    from app.services.ocr import read_image
    from app.services.reputation import default_reputation
    from app.services.vision import embed_images, embed_texts
    from evaluation.datasets import load_caption_pairs
    from evaluation.synth_images import render_login_page, render_sms

    reputation = default_reputation()
    no_claims = lambda text: ([], "insufficient_evidence")  # noqa: E731  isolate image signals

    def verdict(*inputs) -> str:
        case = VerificationCase(case_id="img", created_at="", inputs=list(inputs))
        return getattr(analyze_case(case, text_analyzer=no_claims).verdict, "value", "")

    def screenshot(png: bytes):
        return build_image_input(png, file_name="s.png", mime_type="image/png")

    report: dict[str, object] = {}

    # 1 + 2: SMS screenshots built from the scam cases
    cases = load_scam_cases()
    recalls, gold, predicted = [], [], []
    for case in cases:
        png = render_sms(case["message"])
        wanted = re.findall(r"[a-z0-9]+", case["message"].lower())
        got = set(re.findall(r"[a-z0-9]+", read_image(png).text.lower()))
        recalls.append(sum(word in got for word in wanted) / len(wanted))
        gold.append("deceptive" if case["label"] == "deceptive" else "not_deceptive")
        predicted.append("deceptive" if verdict(screenshot(png)) == "deceptive" else "not_deceptive")
    report["ocr_word_recall"] = {"n": len(recalls), "mean": round(float(np.mean(recalls)), 4), "min": round(min(recalls), 4),
                                 "baseline_trocr_whole_image": 0.0}
    report["sms_screenshot_scam_detection"] = classification_report(gold, predicted, ["deceptive", "not_deceptive"])

    # 3: login-page mockups
    brands = [b for b in ("paypal", "amazon", "microsoft", "apple", "netflix", "google", "github", "dropbox", "linkedin", "instagram")
              if b in reputation.brands]
    results: dict[str, list[str]] = {"lookalike_address": [], "consistent": [], "brand_vs_wrong_link": []}
    for index, brand in enumerate(brands):
        official = sorted(reputation.brands[brand])[0]
        name = brand.capitalize()
        fake = f"{brand}-secure-login.top" if index % 2 == 0 else f"{brand.replace('l', '1', 1)}-verify.click"
        results["lookalike_address"].append(verdict(screenshot(render_login_page(name, f"{fake}/signin"))))
        results["consistent"].append(
            verdict(screenshot(render_login_page(name, f"{official}/signin")), build_url_input(f"https://www.{official}/signin"))
        )
        results["brand_vs_wrong_link"].append(
            verdict(screenshot(render_login_page(name, f"{official}/signin")), build_url_input("https://www.mayoclinic.org/login"))
        )
    report["login_pages"] = {
        group: {label: values.count(label) for label in ("deceptive", "needs_review", "genuine")} | {"n": len(values)}
        for group, values in results.items()
    }

    # 4: CLIP caption consistency on real photos, at the shipped threshold
    pairs = load_caption_pairs(60)
    similarity = embed_images([p["image"] for p in pairs]) @ embed_texts([p["caption"] for p in pairs]).T
    n = len(pairs)
    matched = np.diag(similarity)
    mismatched = similarity[~np.eye(n, dtype=bool)]
    report["clip_caption_consistency"] = {
        "n_pairs": n,
        "top1_retrieval": round(float(np.mean([np.argmax(similarity[i]) == i for i in range(n)])), 4),
        "pairwise_auc": round(float(np.mean([matched[i] > similarity[i, j] for i in range(n) for j in range(n) if i != j])), 4),
        "threshold": CLIP_MISMATCH_THRESHOLD,
        "mismatched_pairs_flagged": round(float(np.mean(mismatched < CLIP_MISMATCH_THRESHOLD)), 4),
        "matched_pairs_wrongly_flagged": round(float(np.mean(matched < CLIP_MISMATCH_THRESHOLD)), 4),
        "note": "threshold was chosen on these same pairs, so the flag rates are optimistic",
    }
    return report


def run_audio() -> dict[str, object]:
    """Speech-to-text accuracy and spoken-scam detection. Word error rate, so lower is better."""
    import time
    from dataclasses import replace

    import numpy as np

    from app.services.audio import WhisperTranscriber
    from app.services.case_analysis import analyze_case
    from app.services.case_store import VerificationCase, build_audio_input
    from evaluation.audio_data import degrade_to_phone_line, ensure_tts_audio, load_librispeech, word_error_rate

    no_claims = lambda text: ([], "insufficient_evidence")  # noqa: E731  isolate the audio signals
    items = ensure_tts_audio()
    scripts = [i for i in items if i["name"].endswith(tuple("0123456789")) and "script" in i["name"]]
    real = load_librispeech(25)
    models = {"base": WhisperTranscriber("base"), "small": WhisperTranscriber("small")}
    report: dict[str, object] = {}

    def wav(item) -> bytes:
        return Path(item["file"]).read_bytes()

    # 1: word error rate, real human speech and phone-line speech, two model sizes
    accuracy = {}
    for name, transcriber in models.items():
        started, audio_seconds, wers = time.time(), 0.0, []
        for clip in real:
            transcript = transcriber.transcribe(clip["audio"])
            audio_seconds += transcript.duration_seconds
            wers.append(word_error_rate(clip["text"], transcript.text))
        real_time_factor = (time.time() - started) / audio_seconds
        phone = [word_error_rate(i["text"], transcriber.transcribe(degrade_to_phone_line(wav(i))).text) for i in scripts]
        clean = [word_error_rate(i["text"], transcriber.transcribe(wav(i)).text) for i in scripts]
        accuracy[name] = {
            "real_speech_wer": round(float(np.mean(wers)), 4),
            "real_speech_n": len(wers),
            "tts_clean_wer": round(float(np.mean(clean)), 4),
            "tts_phone_line_wer": round(float(np.mean(phone)), 4),
            "seconds_to_transcribe_one_second_of_audio": round(real_time_factor, 2),
        }
    report["word_error_rate"] = accuracy

    # 2: spoken-scam detection, audio only, with the default (small) model
    transcriber = models["small"]

    def verdict_for(content: bytes) -> str:
        item = build_audio_input(content, file_name="v.wav", mime_type="audio/wav")
        transcript = transcriber.transcribe(content)
        item = replace(item, text=transcript.text or None, transcription_status="completed", audio_meta=transcript.metadata())
        case = VerificationCase(case_id="audio", created_at="", inputs=[item])
        return getattr(analyze_case(case, text_analyzer=no_claims).verdict, "value", "")

    for label, prepare, subset in (
        ("clean_speech", lambda i: wav(i), items),
        ("phone_line_speech", lambda i: degrade_to_phone_line(wav(i)), scripts),
    ):
        gold, predicted, details = [], [], []
        for item in subset:
            verdict = verdict_for(prepare(item))
            gold.append(item["label"])
            predicted.append("deceptive" if verdict == "deceptive" else "not_deceptive")
            details.append({"name": item["name"], "gold": item["label"], "verdict": verdict})
        result = classification_report(gold, predicted, ["deceptive", "not_deceptive"])
        result["details"] = details
        report[f"scam_detection_{label}"] = result
    return report


def run_ablation() -> dict[str, object]:
    """Add one channel at a time to the multi-channel scam benchmark and measure what each one buys."""
    from dataclasses import replace

    from app.services.audio import WhisperTranscriber
    from app.services.case_analysis import analyze_case
    from app.services.case_store import (
        VerificationCase, build_audio_input, build_image_input, build_text_input, build_url_input,
    )
    from app.services.reputation import default_reputation
    from evaluation.cross_channel_bench import build_cases, synthesise_voices

    reputation = default_reputation()
    cases = build_cases(reputation)
    voices = synthesise_voices(cases)
    transcriber = WhisperTranscriber("small")
    heard = {name: transcriber.transcribe(content) for name, content in voices.items()}
    no_claims = lambda text: ([], "insufficient_evidence")  # noqa: E731  isolate the channel signals

    def build_inputs(case, channels):
        inputs = [build_text_input(case["text"])]
        if "url" in channels and case["url"]:
            inputs.append(build_url_input(case["url"]))
        if "image" in channels and case["image"]:
            inputs.append(build_image_input(case["image"], file_name="s.png", mime_type="image/png"))
        if "audio" in channels and case["name"] in heard:
            t = heard[case["name"]]
            item = build_audio_input(voices[case["name"]], file_name="v.wav", mime_type="audio/wav")
            inputs.append(replace(item, text=t.text or None, transcription_status="completed", audio_meta=t.metadata()))
        return inputs

    configs = [
        ("1 text only", {"text"}, False),
        ("2 + link", {"text", "url"}, False),
        ("3 + screenshot", {"text", "url", "image"}, False),
        ("4 + voice note", {"text", "url", "image", "audio"}, False),
        ("5 + cross-channel reasoning", {"text", "url", "image", "audio"}, True),
    ]
    results = {}
    for label, channels, cross in configs:
        verdicts = {}
        for case in cases:
            run_case = VerificationCase(case_id=case["name"], created_at="", inputs=build_inputs(case, channels))
            verdicts[case["name"]] = getattr(analyze_case(run_case, text_analyzer=no_claims, cross_channel=cross).verdict, "value", "")

        def rate(gold, ok):
            group = [c for c in cases if c["gold"] == gold]
            return round(sum(ok(verdicts[c["name"]]) for c in group) / len(group), 3)

        deceptive = [c for c in cases if c["gold"] == "deceptive"]
        results[label] = {
            "scams_caught_as_deceptive": rate("deceptive", lambda v: v == "deceptive"),
            "scams_at_least_flagged": rate("deceptive", lambda v: v != "genuine"),
            "genuine_cleared": rate("genuine", lambda v: v == "genuine"),
            "genuine_falsely_deceptive": rate("genuine", lambda v: v == "deceptive"),
            "conflicts_not_cleared": rate("not_genuine", lambda v: v != "genuine"),
            "wrong_confident_answers": sum(
                (c["gold"] != "genuine" and verdicts[c["name"]] == "genuine")
                or (c["gold"] == "genuine" and verdicts[c["name"]] == "deceptive")
                for c in cases
            ),
            "per_case": verdicts,
        }
    return {"n_cases": len(cases), "by_channels": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=["phishing_url", "claims", "scam", "image", "audio", "ablation", "all"], default="all")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--use-llm", action="store_true", help="leave the Groq investigator enabled")
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args()

    if not args.use_llm:
        os.environ.pop("GROQ_API_KEY", None)

    suites = ["phishing_url", "claims", "scam", "image", "audio", "ablation"] if args.suite == "all" else [args.suite]
    results: dict[str, object] = {
        "created": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "llm_enabled": args.use_llm,
    }
    for name in suites:
        started = time.time()
        if name == "phishing_url":
            results[name] = run_phishing_url()
        elif name == "claims":
            results[name] = run_claims()
        elif name == "image":
            results[name] = run_image()
        elif name == "audio":
            results[name] = run_audio()
        elif name == "ablation":
            results[name] = run_ablation()
        else:
            results[name] = run_scam(args.mode)
        results[name]["seconds"] = round(time.time() - started, 1)
        _print_summary(name, results[name])

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.out}")
    return 0


def _print_summary(name: str, report: dict[str, object]) -> None:
    if name == "ablation":
        print(f"\n== ablation ({report['n_cases']} cases, {report['seconds']}s) ==")
        for label, row in report["by_channels"].items():
            print(f"  {label:<30}", {k: v for k, v in row.items() if k != "per_case"})
        return
    if name in {"image", "audio"}:
        print(f"\n== {name} ({report['seconds']}s) ==")
        for key, value in report.items():
            if key == "seconds":
                continue
            if isinstance(value, dict) and "details" in value:
                value = {k: v for k, v in value.items() if k not in {"details", "per_label"}}
            print(f"  {key}: {value}")
        return
    print(f"\n== {name} (n={report['n']}, {report['seconds']}s) ==")
    print(f"accuracy={report['accuracy']}  macro_f1={report['macro_f1']}", end="")
    if "abstain_rate" in report:
        print(f"  abstain_rate={report['abstain_rate']}  acc_on_answered={report['accuracy_on_answered']}", end="")
    print()
    for label, scores in report["per_label"].items():
        print(f"  {label:<24} P={scores['precision']:<6} R={scores['recall']:<6} F1={scores['f1']:<6} n={scores['support']}")
    print(f"  confusion: {report['confusion']}")


if __name__ == "__main__":
    sys.exit(main())
