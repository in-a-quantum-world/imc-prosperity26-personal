"""
Discord Intelligence Scraper for IMC Prosperity
================================================
Monitors Discord channels, filters noise/misinformation, and summarises
strategy-relevant information using Claude.

Setup
-----
1. Create a Discord bot at https://discord.com/developers/applications
2. Give it "Read Message History" + "View Channels" permissions
3. Invite it to the IMC Prosperity Discord
4. Set environment variables:
       DISCORD_BOT_TOKEN=your_bot_token
       ANTHROPIC_API_KEY=your_anthropic_key
5. Install dependencies:
       pip install discord.py anthropic

Usage
-----
    python discord_intel.py                   # live monitor mode
    python discord_intel.py --backfill 500    # backfill last 500 msgs per channel
    python discord_intel.py --report          # print latest analysis report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import discord
from discord.ext import tasks

# ─── Config ──────────────────────────────────────────────────────────────────

BOT_TOKEN      = os.environ.get("DISCORD_BOT_TOKEN", "")
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
DB_PATH        = Path(__file__).parent / "discord_intel.db"

# Channels to monitor — fill in channel IDs (right-click channel → Copy ID)
# Leave empty to monitor ALL channels the bot can read.
WATCH_CHANNEL_IDS: list[int] = []

# How many messages to batch before sending to Claude for analysis
ANALYSIS_BATCH_SIZE = 30

# How often (seconds) to run a batch analysis on buffered messages
ANALYSIS_INTERVAL_SECS = 120

# ─── Database ────────────────────────────────────────────────────────────────

def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY,
            channel_id  INTEGER,
            channel_name TEXT,
            author      TEXT,
            content     TEXT,
            timestamp   TEXT,
            analysed    INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT,
            batch_start INTEGER,
            batch_end   INTEGER,
            report      TEXT
        )
    """)
    conn.commit()
    return conn


def save_message(conn: sqlite3.Connection, msg: discord.Message) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO messages
           (id, channel_id, channel_name, author, content, timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            msg.id,
            msg.channel.id,
            getattr(msg.channel, "name", str(msg.channel.id)),
            str(msg.author),
            msg.content,
            msg.created_at.isoformat(),
        ),
    )
    conn.commit()


def get_unanalysed(conn: sqlite3.Connection, limit: int) -> list[dict]:
    rows = conn.execute(
        """SELECT id, channel_name, author, content, timestamp
           FROM messages WHERE analysed = 0
           ORDER BY timestamp ASC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        {"id": r[0], "channel": r[1], "author": r[2], "content": r[3], "ts": r[4]}
        for r in rows
    ]


def mark_analysed(conn: sqlite3.Connection, ids: list[int]) -> None:
    conn.executemany(
        "UPDATE messages SET analysed = 1 WHERE id = ?",
        [(i,) for i in ids],
    )
    conn.commit()


def save_analysis(conn: sqlite3.Connection, ids: list[int], report: str) -> None:
    conn.execute(
        "INSERT INTO analyses (created_at, batch_start, batch_end, report) VALUES (?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), min(ids), max(ids), report),
    )
    conn.commit()


def get_latest_reports(conn: sqlite3.Connection, n: int = 5) -> list[str]:
    rows = conn.execute(
        "SELECT created_at, report FROM analyses ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    return [f"[{r[0]}]\n{r[1]}" for r in reversed(rows)]

# ─── Claude Analysis ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a competitive intelligence analyst for an IMC Prosperity algorithmic
trading competition team. You are reading Discord messages from the competition server.

Your job:
1. Extract genuinely useful strategy/market-structure information (observed price patterns,
   product behaviours, position limits, conversion mechanics, etc.).
2. Flag messages that appear to be deliberate misinformation or designed to mislead other
   teams (e.g. false claims about fair values, fake "discoveries", coordinated narratives).
3. Identify which products/strategies are getting the most attention from other teams.
4. Summarise actionable takeaways for our own trading.

Be skeptical. Teams lie. Weight concrete observations over vague claims.
Output a structured report with sections:
  VERIFIED INTEL, SUSPECTED MISINFORMATION, COMPETITOR FOCUS, ACTIONABLE TAKEAWAYS."""


def analyse_batch(messages: list[dict]) -> str:
    if not ANTHROPIC_KEY:
        return "[ERROR] ANTHROPIC_API_KEY not set."

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    formatted = "\n".join(
        f"[{m['ts'][:16]}] #{m['channel']} {m['author']}: {m['content']}"
        for m in messages
        if m["content"].strip()
    )

    if not formatted.strip():
        return "[No content to analyse]"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Analyse these Discord messages:\n\n{formatted}"}],
    )
    return response.content[0].text

# ─── Discord Bot ─────────────────────────────────────────────────────────────

class IntelBot(discord.Client):
    def __init__(self, conn: sqlite3.Connection, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)
        self.conn = conn
        self._buffer: list[discord.Message] = []

    async def on_ready(self):
        print(f"Logged in as {self.user}. Monitoring {len(WATCH_CHANNEL_IDS) or 'all'} channel(s).")
        self.analysis_loop.start()

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if WATCH_CHANNEL_IDS and message.channel.id not in WATCH_CHANNEL_IDS:
            return
        save_message(self.conn, message)
        print(f"  [{message.channel}] {message.author}: {message.content[:80]}")

    @tasks.loop(seconds=ANALYSIS_INTERVAL_SECS)
    async def analysis_loop(self):
        batch = get_unanalysed(self.conn, ANALYSIS_BATCH_SIZE)
        if not batch:
            return
        print(f"\n--- Analysing {len(batch)} messages ---")
        report = await asyncio.to_thread(analyse_batch, batch)
        ids = [m["id"] for m in batch]
        save_analysis(self.conn, ids, report)
        mark_analysed(self.conn, ids)
        print(report)
        print("--- End of analysis ---\n")


async def backfill(conn: sqlite3.Connection, limit: int):
    """Fetch the last `limit` messages from each watched channel."""
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        total = 0
        channels = (
            [client.get_channel(cid) for cid in WATCH_CHANNEL_IDS]
            if WATCH_CHANNEL_IDS
            else [ch for g in client.guilds for ch in g.text_channels]
        )
        for ch in channels:
            if ch is None:
                continue
            try:
                async for msg in ch.history(limit=limit, oldest_first=False):
                    if not msg.author.bot:
                        save_message(conn, msg)
                        total += 1
                print(f"  Backfilled #{ch.name}: up to {limit} messages")
            except discord.Forbidden:
                print(f"  No access to #{ch.name}, skipping")
        print(f"Backfill complete. {total} messages saved.")
        await client.close()

    await client.start(BOT_TOKEN)

# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Discord Intel Tool for IMC Prosperity")
    parser.add_argument("--backfill", type=int, metavar="N",
                        help="Backfill last N messages per channel then exit")
    parser.add_argument("--report", action="store_true",
                        help="Print latest analysis reports and exit")
    parser.add_argument("--analyse-now", action="store_true",
                        help="Run analysis on unanalysed messages immediately and exit")
    args = parser.parse_args()

    conn = init_db(DB_PATH)

    if args.report:
        reports = get_latest_reports(conn, n=5)
        if not reports:
            print("No analyses yet.")
        for r in reports:
            print(r)
            print("=" * 60)
        return

    if not BOT_TOKEN:
        print("ERROR: Set DISCORD_BOT_TOKEN environment variable.")
        return

    if args.backfill:
        asyncio.run(backfill(conn, args.backfill))
        return

    if args.analyse_now:
        batch = get_unanalysed(conn, ANALYSIS_BATCH_SIZE)
        if not batch:
            print("No unanalysed messages.")
            return
        report = analyse_batch(batch)
        ids = [m["id"] for m in batch]
        save_analysis(conn, ids, report)
        mark_analysed(conn, ids)
        print(report)
        return

    # Live monitor mode
    bot = IntelBot(conn)
    bot.run(BOT_TOKEN)


if __name__ == "__main__":
    main()
