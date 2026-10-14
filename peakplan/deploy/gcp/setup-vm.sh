#!/usr/bin/env bash
# Runs ON the VM as root. Idempotent: safe to re-run for updates.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
APP=/opt/peakplan/server
DATA=/var/lib/peakplan

# 1 GB swap: e2-micro has 1 GB RAM, npm install needs headroom.
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

apt-get update -qq
apt-get install -y -qq curl ca-certificates sqlite3 python3 >/dev/null

# Node.js 22 (needs >= 22.13 for the built-in SQLite)
if ! command -v node >/dev/null || ! node -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>22||(a==22&&b>=13)?0:1)'; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
  apt-get install -y -qq nodejs >/dev/null
fi

id peakplan >/dev/null 2>&1 || useradd --system --home "$DATA" --shell /usr/sbin/nologin peakplan
mkdir -p "$APP" "$DATA/backups"

# Token: keep the existing one unless this is the first install.
if [ ! -f /etc/peakplan.env ] && [ -f /tmp/peakplan.env ]; then
  { grep '^PEAKPLAN_TOKEN=' /tmp/peakplan.env; echo "PEAKPLAN_DB=$DATA/peakplan.db"; echo "PORT=8787"; } > /etc/peakplan.env
  chmod 600 /etc/peakplan.env
fi
rm -f /tmp/peakplan.env

systemctl stop peakplan 2>/dev/null || true

# Database from the Mac: only on first install, never overwrites server data.
if [ -f /tmp/peakplan.db ] && [ ! -f "$DATA/peakplan.db" ]; then
  mv /tmp/peakplan.db "$DATA/peakplan.db"
  echo "==> Imported your existing database"
fi
rm -f /tmp/peakplan.db
chown -R peakplan:peakplan "$DATA"

echo "==> Building server"
rm -rf "$APP/src" "$APP/test"
tar -xzf /tmp/server.tgz -C "$APP"
rm -f /tmp/server.tgz
cd "$APP"
npm ci --no-audit --no-fund --loglevel=error
npm run build --silent
npm prune --omit=dev --no-audit --no-fund --loglevel=error

cat > /etc/systemd/system/peakplan.service <<UNIT
[Unit]
Description=PeakPlan server
After=network-online.target
Wants=network-online.target

[Service]
User=peakplan
EnvironmentFile=/etc/peakplan.env
WorkingDirectory=$APP
ExecStart=/usr/bin/node --disable-warning=ExperimentalWarning $APP/dist/index.js
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now peakplan >/dev/null
sleep 2
curl -fsS localhost:8787/health >/dev/null && echo "==> Server running" || { journalctl -u peakplan -n 30 --no-pager; exit 1; }

# Daily backup on the VM's own disk, last 14 days (snapshots would cost money).
cat > /etc/cron.daily/peakplan-backup <<'CRON'
#!/bin/sh
sqlite3 /var/lib/peakplan/peakplan.db ".backup '/var/lib/peakplan/backups/peakplan-$(date +%Y%m%d).db'"
find /var/lib/peakplan/backups -name 'peakplan-*.db' -mtime +14 -delete
CRON
chmod +x /etc/cron.daily/peakplan-backup

# If HTTPS is served by Caddy (enable-https.sh), Tailscale isn't needed.
if [ -f /etc/caddy/Caddyfile ] && grep -q sslip.io /etc/caddy/Caddyfile; then
  echo "==> Server URL: https://$(head -1 /etc/caddy/Caddyfile | cut -d' ' -f1)"
  exit 0
fi

# Public HTTPS address via Tailscale Funnel: fixed URL, no open ports, no domain needed.
if ! command -v tailscale >/dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sh >/dev/null
fi
if ! tailscale status >/dev/null 2>&1; then
  echo
  echo "==> Log in to Tailscale: open the link below on your Mac (create a free account if needed)."
  tailscale up --hostname=peakplan
fi
echo "==> Turning on Tailscale Funnel (if asked, open the link to allow Funnel, then come back)"
tailscale funnel --bg 8787
HOST=$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')
echo
echo "============================================================"
echo " PeakPlan server URL:   https://$HOST"
echo " Claude connector URL:  https://$HOST/mcp/<your token>"
echo "============================================================"
echo
echo "IMPORTANT: in https://login.tailscale.com/admin/machines open 'peakplan' → ⋯ → Disable key expiry,"
echo "otherwise Tailscale logs the server out after 180 days."
