@echo off
REM Run this once after getting your tokens, then run the bot in the same terminal.
REM Double-click this file OR run it from a terminal before launching discord_intel.py

set DISCORD_BOT_TOKEN=PASTE_YOUR_BOT_TOKEN_HERE
set ANTHROPIC_API_KEY=PASTE_YOUR_ANTHROPIC_KEY_HERE

echo Environment variables set.
echo Now run:  python tools\discord_intel.py --backfill 500
