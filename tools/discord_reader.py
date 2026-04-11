"""
discord_reader.py
-----------------
Reads Discord channel exports (from DiscordChatExporter) and displays
messages organised by channel, author, and timestamp.

Usage
-----
    python tools/discord_reader.py --dir exports/
    python tools/discord_reader.py --dir exports/ --channel general
    python tools/discord_reader.py --dir exports/ --user "someuser"
    python tools/discord_reader.py --dir exports/ --since "2026-04-10"
    python tools/discord_reader.py --dir exports/ --search "fair value"

DiscordChatExporter export format
----------------------------------
DiscordChatExporter (https://github.com/Tyrrrz/DiscordChatExporter) exports
each channel as a JSON file. Typical structure:

    exports/
        general.json
        strategy-discussion.json
        round-1.json
        ...

Each JSON file has this shape:
    {
        "guild": { "name": "...", "id": "..." },
        "channel": { "name": "...", "id": "...", "type": "..." },
        "messages": [
            {
                "id": "...",
                "timestamp": "2026-04-10T12:34:56.000+00:00",
                "author": { "name": "...", "id": "...", "isBot": false },
                "content": "...",
                "attachments": [...],
                "embeds": [...]
            },
            ...
        ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# ── Colours (works on Windows 10+ terminals) ──────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
DIM    = "\033[2m"


# ── Loader ────────────────────────────────────────────────────────────────────

def load_exports(export_dir: Path) -> list[dict]:
    """Load all .json export files from a directory. Returns list of channel dicts."""
    files = sorted(export_dir.glob("*.json"))
    if not files:
        print(f"No .json files found in {export_dir}")
        sys.exit(1)

    channels = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            # Support both top-level 'messages' and wrapped format
            if "messages" in data:
                channel_name = data.get("channel", {}).get("name", f.stem)
                guild_name   = data.get("guild", {}).get("name", "Unknown Server")
                channels.append({
                    "file":         f.name,
                    "channel_name": channel_name,
                    "guild_name":   guild_name,
                    "messages":     data["messages"],
                })
            else:
                print(f"  Skipping {f.name} — unexpected format (no 'messages' key)")
        except json.JSONDecodeError as e:
            print(f"  Skipping {f.name} — JSON error: {e}")

    return channels


# ── Message normalisation ─────────────────────────────────────────────────────

def parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp, handling various formats DiscordChatExporter uses."""
    ts_str = ts_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        # Fallback: strip microseconds if needed
        return datetime.fromisoformat(ts_str[:19] + "+00:00")


def normalise(msg: dict, channel_name: str) -> dict:
    author = msg.get("author", {})
    return {
        "id":           msg.get("id", ""),
        "channel":      channel_name,
        "timestamp":    parse_ts(msg["timestamp"]),
        "author_name":  author.get("name", author.get("nickname", "Unknown")),
        "author_id":    author.get("id", ""),
        "is_bot":       author.get("isBot", False),
        "content":      msg.get("content", "").strip(),
        "attachments":  [a.get("url", "") for a in msg.get("attachments", [])],
    }


# ── Filtering ─────────────────────────────────────────────────────────────────

def apply_filters(
    messages: list[dict],
    channel:  str | None,
    user:     str | None,
    since:    datetime | None,
    search:   str | None,
    no_bots:  bool,
) -> list[dict]:
    out = messages
    if channel:
        out = [m for m in out if channel.lower() in m["channel"].lower()]
    if user:
        out = [m for m in out if user.lower() in m["author_name"].lower()]
    if since:
        out = [m for m in out if m["timestamp"] >= since]
    if search:
        out = [m for m in out if search.lower() in m["content"].lower()]
    if no_bots:
        out = [m for m in out if not m["is_bot"]]
    return out


# ── Display ───────────────────────────────────────────────────────────────────

def print_message(m: dict, show_channel: bool = True) -> None:
    ts      = m["timestamp"].strftime("%Y-%m-%d %H:%M")
    channel = f"{CYAN}#{m['channel']}{RESET} " if show_channel else ""
    author  = f"{YELLOW}{m['author_name']}{RESET}"
    content = m["content"] if m["content"] else DIM + "[no text content]" + RESET
    attachments = ""
    if m["attachments"]:
        attachments = f"\n  {DIM}[attachment: {m['attachments'][0]}]{RESET}"
    print(f"{DIM}{ts}{RESET} {channel}{author}: {content}{attachments}")


def print_summary(channels: list[dict], all_messages: list[dict]) -> None:
    print(f"\n{BOLD}=== Discord Export Summary ==={RESET}")
    print(f"  Channels loaded : {len(channels)}")
    print(f"  Total messages  : {len(all_messages)}")

    if all_messages:
        timestamps = [m["timestamp"] for m in all_messages]
        print(f"  Earliest        : {min(timestamps).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Latest          : {max(timestamps).strftime('%Y-%m-%d %H:%M UTC')}")

    print(f"\n{BOLD}Channels:{RESET}")
    for ch in channels:
        count = sum(1 for m in all_messages if m["channel"] == ch["channel_name"])
        print(f"  #{ch['channel_name']:<30} {count:>5} messages")

    # Top authors
    from collections import Counter
    author_counts = Counter(m["author_name"] for m in all_messages if not m["is_bot"])
    print(f"\n{BOLD}Top 10 most active users:{RESET}")
    for author, count in author_counts.most_common(10):
        print(f"  {YELLOW}{author:<30}{RESET} {count:>5} messages")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Read and explore DiscordChatExporter JSON exports"
    )
    parser.add_argument(
        "--dir", required=True, metavar="PATH",
        help="Directory containing exported .json files"
    )
    parser.add_argument("--channel", metavar="NAME",  help="Filter to channels matching NAME")
    parser.add_argument("--user",    metavar="NAME",  help="Filter to messages from USER")
    parser.add_argument("--since",   metavar="DATE",  help="Only show messages from DATE onwards (YYYY-MM-DD)")
    parser.add_argument("--search",  metavar="TERM",  help="Only show messages containing TERM")
    parser.add_argument("--no-bots", action="store_true", help="Exclude bot messages")
    parser.add_argument("--summary", action="store_true", help="Show summary stats only, no messages")
    parser.add_argument("--limit",   type=int, default=200, metavar="N",
                        help="Max messages to print (default 200, 0 = unlimited)")
    args = parser.parse_args()

    export_dir = Path(args.dir)
    if not export_dir.exists():
        print(f"Directory not found: {export_dir}")
        sys.exit(1)

    # Load
    channels = load_exports(export_dir)
    all_messages: list[dict] = []
    for ch in channels:
        for raw in ch["messages"]:
            try:
                all_messages.append(normalise(raw, ch["channel_name"]))
            except Exception:
                continue

    # Sort chronologically
    all_messages.sort(key=lambda m: m["timestamp"])

    # Summary always shown
    print_summary(channels, all_messages)

    if args.summary:
        return

    # Parse --since
    since_dt = None
    if args.since:
        since_dt = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    # Filter
    filtered = apply_filters(
        all_messages,
        channel = args.channel,
        user    = args.user,
        since   = since_dt,
        search  = args.search,
        no_bots = args.no_bots,
    )

    if not filtered:
        print("No messages match your filters.")
        return

    limit = args.limit if args.limit > 0 else len(filtered)
    showing = filtered[-limit:]  # most recent N if limiting

    print(f"{BOLD}Showing {len(showing)} of {len(filtered)} matching messages:{RESET}\n")
    prev_channel = None
    for m in showing:
        # Print channel header when it changes
        if m["channel"] != prev_channel:
            print(f"\n{BOLD}{CYAN}── #{m['channel']} ──{RESET}")
            prev_channel = m["channel"]
        print_message(m, show_channel=False)


if __name__ == "__main__":
    main()
