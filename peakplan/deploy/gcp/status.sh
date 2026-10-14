#!/usr/bin/env bash
# Quick health report of the server, run from your Mac.
set -euo pipefail
cd "$(dirname "$0")"
. ./.config
gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --quiet --command '
echo "=== Services";  systemctl is-active peakplan caddy 2>/dev/null | paste -sd" " | sed "s/^/peakplan, caddy: /"
echo "=== Health";    curl -s localhost:8787/health; echo
echo "=== Uptime";    uptime
echo "=== Memory";    free -h | head -2
echo "=== Disk";      df -h / | tail -1
echo "=== Backups";   ls -lh /var/lib/peakplan/backups 2>/dev/null | tail -3
echo "=== Last server log lines"; sudo journalctl -u peakplan -n 10 --no-pager -o cat
'
