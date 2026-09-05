#!/bin/bash
cd ~/Bot\ Fake\ HTool
export PATH="$HOME/Bot Fake HTool/venv/bin:$PATH"
kill $(lsof -t -i:8080) 2>/dev/null
pkill -f bot_orchestrator 2>/dev/null
sleep 2
nohup "$HOME/Bot Fake HTool/venv/bin/uvicorn" backend_api:app --host 0.0.0.0 --port 8080 > api.log 2>&1 &
sleep 3
nohup "$HOME/Bot Fake HTool/venv/bin/python" bot_orchestrator.py > bot.log 2>&1 &
sleep 3
echo "=== API ===" && tail -5 api.log
echo "=== BOT ===" && tail -5 bot.log
