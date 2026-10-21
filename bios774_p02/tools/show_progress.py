"""show_progress.py — live, readable view of a headless Claude Code run.

Reads Claude Code's `--output-format stream-json` from stdin, saves every
event to the transcript file, and prints a short line per agent step.
Used by run_agent.sh; not called by the agent.
"""
import json
import sys


def short(s, n=160):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    transcript = open(sys.argv[1], "a")
    for line in sys.stdin:
        transcript.write(line)
        transcript.flush()
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            print(line.rstrip())
            continue
        t = ev.get("type")
        if t == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "text" and block.get("text", "").strip():
                    print(f"\n[agent] {short(block['text'], 400)}", flush=True)
                elif block.get("type") == "tool_use":
                    inp = block.get("input", {})
                    what = inp.get("command") or inp.get("file_path") or inp.get("pattern") or json.dumps(inp)
                    print(f"  → {block.get('name')}: {short(what)}", flush=True)
        elif t == "result":
            print("\n" + "=" * 70)
            print(f"[done] turns={ev.get('num_turns')} time={round(ev.get('duration_ms', 0) / 60000, 1)} min "
                  f"cost=${ev.get('total_cost_usd', 0):.2f} error={ev.get('is_error')}")
            print(ev.get("result", ""))
    transcript.close()


if __name__ == "__main__":
    main()
