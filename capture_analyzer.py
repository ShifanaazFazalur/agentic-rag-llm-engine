"""
WiFi Capture Analyzer
======================
Runs targeted Wireshark filters against a WiFi packet capture (.pcap/.pcapng)
to flag common signs of malicious or anomalous wireless activity — no ML
model required, just Wireshark's own protocol dissection.

Detects:
  - Deauthentication / disassociation floods
  - Probe request floods (recon/scanning behavior)
  - Abnormal retry rate
  - Malformed packets

Requires Wireshark to be installed (its command-line component must be on PATH).

Usage:
  python capture_analyzer.py path/to/capture.pcap
"""

import os
import shutil
import subprocess
import sys
from collections import defaultdict

WIRESHARK_CLI = "tshark"  # Wireshark's command-line component


def check_prerequisites(pcap_path: str) -> None:
    """Fail loudly instead of silently returning zeros for everything."""
    if shutil.which(WIRESHARK_CLI) is None:
        raise RuntimeError(
            "Wireshark not found on PATH. Install Wireshark (https://www.wireshark.org) "
            "and make sure it's on your PATH."
        )
    if not os.path.isfile(pcap_path):
        raise FileNotFoundError(
            f"No capture file found at '{pcap_path}'. Point this at a real .pcap/.pcapng "
            f"file saved from Wireshark, not a placeholder path."
        )


def run_wireshark_filter(pcap_path: str, display_filter: str, fields: list = None) -> list:
    """Run a Wireshark filter against the capture, optionally extracting specific
    fields. Returns one line of output per matching packet."""
    cmd = [WIRESHARK_CLI, "-r", pcap_path]
    if display_filter:
        cmd += ["-Y", display_filter]
    if fields:
        # occurrence=f: when a frame repeats a field (e.g. multiple SSID tags),
        # only take the first value instead of comma-joining them into one string.
        cmd += ["-T", "fields", "-E", "occurrence=f"]
        for f in fields:
            cmd += ["-e", f]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [warning] Wireshark error on filter '{display_filter}': {result.stderr.strip()}")
    return [l for l in result.stdout.splitlines() if l.strip()]


def count_total_packets(pcap_path: str) -> int:
    return len(run_wireshark_filter(pcap_path, ""))


def count_deauth_disassoc(pcap_path: str) -> dict:
    deauth = run_wireshark_filter(pcap_path, "wlan.fc.type_subtype==0x0c")
    disassoc = run_wireshark_filter(pcap_path, "wlan.fc.type_subtype==0x0a")
    return {"deauth_frames": len(deauth), "disassoc_frames": len(disassoc)}


def detect_probe_floods(pcap_path: str, threshold: int = 50) -> dict:
    """Flag single devices sending an unusually high number of probe requests."""
    lines = run_wireshark_filter(pcap_path, "wlan.fc.type_subtype==0x04", fields=["wlan.sa"])
    counts = defaultdict(int)
    for line in lines:
        mac = line.strip()
        if mac:
            counts[mac] += 1
    flooders = {mac: c for mac, c in counts.items() if c >= threshold}
    return {"probe_flood_sources": flooders}


def retry_rate(pcap_path: str) -> dict:
    total = run_wireshark_filter(pcap_path, "wlan.fc.type==2")  # data frames
    retries = run_wireshark_filter(pcap_path, "wlan.fc.type==2 && wlan.fc.retry==1")
    total_n, retry_n = len(total), len(retries)
    pct = round((retry_n / total_n) * 100, 1) if total_n else 0.0
    return {"data_frames": total_n, "retried_frames": retry_n, "retry_rate_pct": pct}


def malformed_packets(pcap_path: str, total_packets: int) -> dict:
    lines = run_wireshark_filter(pcap_path, "_ws.malformed")
    count = len(lines)
    pct = round((count / total_packets) * 100, 1) if total_packets else 0.0
    return {"malformed_packets": count, "malformed_pct": pct}


def analyze_capture(pcap_path: str) -> dict:
    """Run all checks and return a combined findings dict."""
    check_prerequisites(pcap_path)
    total_packets = count_total_packets(pcap_path)
    findings = {"total_packets": total_packets}
    findings.update(count_deauth_disassoc(pcap_path))
    findings.update(detect_probe_floods(pcap_path))
    findings.update(retry_rate(pcap_path))
    findings.update(malformed_packets(pcap_path, total_packets))
    return findings


def format_findings(findings: dict) -> str:
    """Turn raw findings into a short plain-text summary (for humans and the LLM)."""
    return "\n".join([
        f"- Total packets in capture: {findings['total_packets']}",
        f"- Deauthentication frames: {findings['deauth_frames']}",
        f"- Disassociation frames: {findings['disassoc_frames']}",
        f"- Probe flood sources (>=50 probes from one device): {findings['probe_flood_sources'] or 'none'}",
        f"- Retry rate: {findings['retry_rate_pct']}% ({findings['retried_frames']}/{findings['data_frames']} data frames)",
        f"- Malformed packets: {findings['malformed_packets']} ({findings['malformed_pct']}% of total capture — "
        f"note: passive/promiscuous captures often show elevated malformed rates from RF noise, this alone isn't "
        f"necessarily an attack indicator)",
    ])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python capture_analyzer.py path/to/capture.pcap")
        sys.exit(1)

    pcap_path = sys.argv[1]
    print(f"Analyzing {pcap_path}...\n")
    print(format_findings(analyze_capture(pcap_path)))
