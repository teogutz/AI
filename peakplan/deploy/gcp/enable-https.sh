#!/usr/bin/env bash
# Gives the server a public HTTPS address without Tailscale:
#   https://<ip-with-dashes>.sslip.io  (free DNS name that resolves to the VM's IP)
# Caddy obtains and renews a free Let's Encrypt certificate automatically.
# Opens ports 80/443 on the VM (firewall rules are free).
set -euo pipefail
cd "$(dirname "$0")"
. ./.config

echo "==> Opening ports 80 and 443"
if ! gcloud compute firewall-rules describe peakplan-web --project "$PROJECT" >/dev/null 2>&1; then
  gcloud compute firewall-rules create peakplan-web --project "$PROJECT" \
    --allow=tcp:80,tcp:443 --target-tags=peakplan-web --direction=INGRESS --quiet >/dev/null
fi
gcloud compute instances add-tags "$VM" --tags=peakplan-web --zone "$ZONE" --project "$PROJECT" --quiet >/dev/null

IP=$(gcloud compute instances describe "$VM" --zone "$ZONE" --project "$PROJECT" \
  --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
HOST="$(echo "$IP" | tr . -).sslip.io"
echo "==> Address: https://$HOST"

gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --quiet --command "sudo bash -s" <<REMOTE
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# Tailscale Funnel/Serve would hold port 443.
if command -v tailscale >/dev/null; then tailscale funnel reset 2>/dev/null || true; tailscale serve reset 2>/dev/null || true; fi
command -v caddy >/dev/null || { apt-get update -qq; apt-get install -y -qq caddy >/dev/null; }
cat > /etc/caddy/Caddyfile <<CADDY
$HOST {
  reverse_proxy localhost:8787
}
CADDY
systemctl enable caddy >/dev/null
systemctl restart caddy
REMOTE

echo "==> Waiting for the HTTPS certificate"
for i in $(seq 1 30); do
  if curl -fsS "https://$HOST/health" >/dev/null 2>&1; then
    echo
    echo "============================================================"
    echo " PeakPlan server URL:   https://$HOST"
    echo " Claude connector URL:  https://$HOST/mcp/<your token>"
    echo "============================================================"
    echo "Note: don't use 'Stop' on the VM in the Google console. Stopping can change its IP and thus this address."
    exit 0
  fi
  sleep 5
done
echo "!! Not reachable yet. Try in a minute:  curl https://$HOST/health"
