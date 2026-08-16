# WiFi Anomaly Digest Agent

An LLM agent that analyzes WiFi packet captures for signs of anomalous or malicious activity — deauthentication floods, ARP spoofing, probe request floods, abnormal retry rates, and malformed packets — and generates a plain-language risk digest using a locally-hosted LLM (Llama 3.2 via Ollama).

## How it works

1. **Analyze** — runs targeted `tshark` filters against a `.pcap`/`.pcapng` capture to flag suspicious patterns (no ML model or training data required)
2. **Score** — computes a risk level (Low/Medium/High) using deterministic, defined thresholds (see below), so the result is consistent and explainable rather than left up to the LLM to decide
3. **Explain** — passes the findings and the specific risk reasons to Llama 3.2 (via Ollama) to generate a plain-English explanation of what's happening
4. **Save** — writes a structured markdown digest, including *why* the risk level was assigned

## Risk level definitions

- **High** — more than 20 deauthentication frames, OR any ARP conflict detected (same IP address claimed by more than one device)
- **Medium** — a device sending a probe flood (50+ probe requests), OR a retry rate over 20%, OR any deauthentication frames present (below the High threshold)
- **Low** — none of the above

## Detects

- Deauthentication / disassociation floods
- ARP spoofing (same IP claimed by multiple MAC addresses)
- Probe request floods (recon/scanning behavior)
- Abnormal retry rate
- Malformed packets (shown as a % of total capture, with context — passive/promiscuous captures naturally show elevated malformed rates from RF noise, so this alone isn't treated as an attack signal)

## Tech stack

Python, tshark (Wireshark CLI), Ollama (Llama 3.2)

## Setup

```bash
git clone https://github.com/ShifanaazFazalur/agentic-rag-llm-engine.git
cd agentic-rag-llm-engine
pip install -r requirements.txt
python agent.py path/to/capture.pcap
```

Requires [Wireshark](https://www.wireshark.org) (for `tshark`) and [Ollama](https://ollama.com) running locally with `ollama pull llama3.2`.

## Example output

Running the agent against a real capture produces a digest like this:

```markdown
# 📡 WiFi Anomaly Digest
**August 15, 2026** · Capture: `TPLink_ArcherB3600_SnifferCh6.pcapng`

## Risk Level: 🟡 Medium

**Why:**
- 3 device(s) sending probe floods (threshold: >=50 probes)
- 13 deauthentication frame(s) present (below High threshold of 20)

The network has been experiencing a moderate level of security risk. Three devices
are sending probe floods exceeding the threshold of 50 probes, which can be used by
attackers to gather information about the network's layout. Additionally, 13
deauthentication frames were detected, indicating possible malicious activity or
interference. These signals suggest the network may warrant closer monitoring.

## Raw Findings

- Total packets in capture: 62916
- Deauthentication frames: 13
- Disassociation frames: 109
- ARP conflicts (same IP, multiple MACs): none
- Probe flood sources (>=50 probes from one device): {'40:e2:30:51:df:99': 565, ...}
- Retry rate: 14.6% (1059/7234 data frames)
- Malformed packets: 10896 (17.3% of total capture)
```

Each run saves a new digest as `digest_YYYY-MM-DD.md` in the project folder.

## Notes

This tool is meant to flag patterns worth a closer look, not to serve as a definitive intrusion detection system — thresholds are simple, defined rules rather than a trained model, and results should be interpreted alongside the raw findings.
