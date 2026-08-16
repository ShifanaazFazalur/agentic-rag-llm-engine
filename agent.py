"""
WiFi Anomaly Digest Agent
===========================
Analyzes a WiFi packet capture for signs of malicious or anomalous activity
(deauth floods, rogue APs, ARP spoofing, probe floods, high retry rates,
malformed packets), then uses a local LLM to explain the findings and rate
the risk level — producing a structured markdown digest.

Pipeline:
  1. Analyze  — run targeted tshark filters against the capture (capture_analyzer.py)
  2. Explain  — send the findings summary to a local LLM (Ollama/Llama 3.2)
  3. Save     — write the digest to markdown

Requirements:
  pip install -r requirements.txt
  Install Wireshark (for tshark): https://www.wireshark.org
  Install Ollama: https://ollama.com -> then: ollama pull llama3.2

Usage:
  python agent.py path/to/capture.pcap
"""

import sys
from pathlib import Path
from datetime import datetime

import ollama

from capture_analyzer import analyze_capture, format_findings


# --- Config ------------------------------------------------------------------

LLM_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are a wireless network security assistant. You will be given
a summary of findings from a WiFi packet capture, along with a risk level that has
already been determined. Write a 2-4 sentence plain-English explanation of what's
happening on the network and why it was rated at that risk level. Do not restate
the risk level as a heading — just explain. Do not use markdown formatting."""


# --- Step 1b: Deterministic risk scoring ---------------------------------------
# Risk is computed from the findings directly, not left up to the LLM to decide —
# this keeps it consistent, doesn't depend on the model following an exact output
# format, and — critically — every level comes with the exact reason it was
# triggered, so "High" or "Medium" is never an unexplained label.
#
# Defined thresholds:
#   HIGH   — more than 20 deauth frames, OR any ARP conflict
#   MEDIUM — a probe-flood source, OR retry rate over 20%, OR any deauth frames at all
#   LOW    — none of the above

def compute_risk_level(findings: dict) -> dict:
    """Returns {"level": "High"/"Medium"/"Low", "reasons": [list of triggered reasons]}."""
    reasons = []

    if findings["deauth_frames"] > 20:
        reasons.append(f"{findings['deauth_frames']} deauthentication frames (threshold: >20)")
    if findings["arp_conflicts"]:
        reasons.append(f"{len(findings['arp_conflicts'])} ARP conflict(s) detected")
    if reasons:
        return {"level": "High", "reasons": reasons}

    if findings["probe_flood_sources"]:
        reasons.append(f"{len(findings['probe_flood_sources'])} device(s) sending probe floods (threshold: >=50 probes)")
    if findings["retry_rate_pct"] > 20:
        reasons.append(f"retry rate of {findings['retry_rate_pct']}% (threshold: >20%)")
    if findings["deauth_frames"] > 0:
        reasons.append(f"{findings['deauth_frames']} deauthentication frame(s) present (below High threshold of 20)")
    if reasons:
        return {"level": "Medium", "reasons": reasons}

    return {"level": "Low", "reasons": ["no deauth activity, rogue APs, ARP conflicts, probe floods, or elevated retry rate"]}


# --- Step 2: Explain findings (LLM) -------------------------------------------

def explain_findings(findings_summary: str, risk: dict) -> str:
    reasons_text = "; ".join(risk["reasons"])
    prompt = (
        f"Risk level already determined: {risk['level']}\n"
        f"Reasons: {reasons_text}\n\n"
        f"Findings:\n{findings_summary}"
    )
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"(LLM unavailable: {e})"


# --- Step 3: Save Digest -------------------------------------------------------

RISK_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

def save_digest(pcap_path: str, findings: dict, verdict: dict) -> str:
    date_str = datetime.now().strftime("%B %d, %Y")
    output_path = f"digest_{datetime.now().strftime('%Y-%m-%d')}.md"
    risk = verdict["risk"]
    emoji = RISK_EMOJI.get(risk["level"], "🟡")
    reasons_list = "\n".join(f"- {r}" for r in risk["reasons"])

    lines = [
        "# 📡 WiFi Anomaly Digest",
        f"**{date_str}** · Capture: `{pcap_path}`\n",
        f"## Risk Level: {emoji} {risk['level']}\n",
        "**Why:**",
        reasons_list + "\n",
        f"{verdict['explanation']}\n",
        "## Raw Findings\n",
        format_findings(findings),
    ]

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path


# --- Agent Orchestrator ---------------------------------------------------------

def run_agent(pcap_path: str):
    print("\n" + "=" * 50)
    print("  WiFi Anomaly Digest Agent")
    print("=" * 50 + "\n")

    print("[ Step 1 ] Analyzing capture...")
    findings = analyze_capture(pcap_path)
    print(format_findings(findings))
    print()

    print("[ Step 2 ] Scoring risk and generating explanation with LLM...")
    risk = compute_risk_level(findings)
    explanation = explain_findings(format_findings(findings), risk)
    verdict = {"risk": risk, "explanation": explanation}
    print(f"  Risk: {risk['level']}")
    for r in risk["reasons"]:
        print(f"    - {r}")
    print()

    print("[ Step 3 ] Saving digest...")
    output_path = save_digest(pcap_path, findings, verdict)
    print(f"  Saved: {output_path}\n")

    print("=" * 50)
    emoji = RISK_EMOJI.get(risk["level"], "🟡")
    print(f"  {emoji} Risk: {risk['level']}")
    print(f"\nDigest saved to: {output_path}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py path/to/capture.pcap")
        sys.exit(1)
    run_agent(sys.argv[1])
