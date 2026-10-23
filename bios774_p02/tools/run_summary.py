"""run_summary.py — render a run's decision log and exact command history as
report-ready Markdown, so the report never has to retype them by hand.

  python tools/run_summary.py --out outputs/<run_id>

Writes:
  <out>/decision_path.md : one short row per decision (Step | Decision | Why),
                       from the plain-language --short/--why fields — for the
                       report's "Decision path" section.
  <out>/decisions.md : every entry of decisions.jsonl as a numbered list
                       (decision / evidence / alternatives rejected / skill)
                       — for the report's appendix.
  <out>/commands.md  : every tools/*.py command the agent ran, in order,
                       taken verbatim from agent_transcript.jsonl.
Paste (or include) both files' content in the report's Decisions and
Reproduce sections.
"""
import argparse
import json
import re
from pathlib import Path


STEP_LABEL = {"objective": "Goal", "preprocessing": "Data prep", "model_selection": "Model",
              "hyperparameters": "Settings", "training_check": "Training check", "sampling": "Generate",
              "evaluation": "Evaluate", "iteration": "Adjust", "recommendation": "Recommend", "other": "Other"}

KEEP = ("inspect_data.py", "train.py", "sample.py", "score.py", "evaluate.py")


def expand(text, env):
    for k, v in sorted(env.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(r"\$\{?" + k + r"\b\}?", lambda _m: v, text)
    return text


def extract_tool_calls(blob, env):
    """Pull the individual tools/*.py invocations out of a compound shell command,
    expanding simple VAR=value assignments and dropping pipes/redirections, --help
    calls, smoke tests and bookkeeping tools."""
    out = []
    # drop heredoc bodies (the agent writing files such as the report), keep the rest
    blob = re.sub(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n.*?\n\1[ \t]*(\n|$)", "\n", blob, flags=re.S)
    blob = blob.replace("\\\n", " ")
    parts = re.split(r"\n|;|&&", blob)
    for part in parts:
        part = part.strip()
        m = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(\S+)", part)
        if m:
            env[m.group(1)] = expand(m.group(2).strip("'\""), env)
            continue
        if not any(k in part for k in KEEP) or "--help" in part or "--smoke" in part:
            continue
        if re.match(r"(do\s+)?(until|while|pgrep|pkill|ps|wait|sleep|tail|cat|ls|kill)\b", part):
            continue                                  # waiting / monitoring, not a tool call
        if not re.search(r"(^|[\s/])python3?\s+(-\S+\s+)*\S*tools/(" + "|".join(k[:-3] for k in KEEP) + r")\.py\b", part):
            continue                                  # mentions a tool but does not run it
        part = re.split(r"\s\|\s|\s2>&1|\s>\s?/dev/null", part)[0].strip()
        if part.startswith("do "):
            part = part[3:].strip()
        part = expand(part, env)
        if re.search(r"\$\{?[A-Za-z_]", part):     # unresolved shell variable (e.g. a loop): not reproducible verbatim
            part = "# (not reproducible verbatim — used a shell variable) " + part
        part = re.sub(r"^\S*python3?\s", "python3 ", part)
        out.append(part)
    return out


def cell(t):
    return " ".join(str(t).split()).replace("|", "/")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)

    lines, rows, n_dec = [], [], 0
    dec = out / "decisions.jsonl"
    if dec.exists():
        for i, l in enumerate(open(dec), 1):
            n_dec = i
            d = json.loads(l)
            label = STEP_LABEL.get(d.get("step", ""), d.get("step", ""))
            short = d.get("short") or d.get("decision", "")
            if not d.get("short") and len(short) > 80:
                short = short[:77].rsplit(" ", 1)[0] + "..."
            rows.append(f"| {i} | {label} | {cell(short)} | {cell(d.get('why', ''))} |")
            skill = f" *({d['skill']} skill)*" if d.get("skill") else ""
            lines.append(f"{i}. **{label}**{skill} — {d.get('decision', '')}")
            if d.get("evidence"):
                lines.append(f"   - *Evidence:* {d['evidence']}")
            if d.get("alternatives_rejected"):
                lines.append(f"   - *Alternatives rejected:* {d['alternatives_rejected']}")
    (out / "decisions.md").write_text("\n".join(lines) + "\n")
    # dash counts set pandoc's relative column widths: narrow #/Step, wide Decision/Why
    table = ["| # | Step | Decision | Why |", "|--|------|------------------|-------------------------|"] + rows
    (out / "decision_path.md").write_text("\n".join(table) + "\n")

    cmds, env = [], {}          # shell variables persist across the agent's calls
    tr = out / "agent_transcript.jsonl"
    if tr.exists():
        for l in open(tr):
            try:
                ev = json.loads(l)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "assistant":
                continue
            for b in ev.get("message", {}).get("content", []):
                if b.get("type") == "tool_use" and b.get("name") == "Bash":
                    cmds += extract_tool_calls(b.get("input", {}).get("command", ""), env)
    body = ("Tool calls in the order the agent ran them (smoke tests and --help omitted):\n\n```bash\n"
            + "\n".join(cmds) + "\n```\n") if cmds else "(no tool commands found in transcript)\n"
    (out / "commands.md").write_text(body)
    print(f"[run_summary] {n_dec} decisions -> {out / 'decision_path.md'}, {out / 'decisions.md'}; "
          f"{len(cmds)} commands -> {out / 'commands.md'}")


if __name__ == "__main__":
    main()
