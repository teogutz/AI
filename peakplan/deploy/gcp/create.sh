#!/usr/bin/env bash
# Creates the PeakPlan server on Google Cloud's Always Free tier and moves your data there.
# Run from your Mac:  ./create.sh            (add --with-data to copy server/data/peakplan.db)
#
# Free-tier settings used (do not change unless you accept charges):
#   e2-micro VM · us-central1 · 30 GB *standard* persistent disk · Debian 12 · no snapshots
set -euo pipefail
cd "$(dirname "$0")"
SERVER_DIR="../../server"
ZONE="us-central1-a"
VM="peakplan"
WITH_DATA=0
[ "${1:-}" = "--with-data" ] && WITH_DATA=1

command -v gcloud >/dev/null || { echo "Install the Google Cloud CLI first:  brew install --cask google-cloud-sdk"; exit 1; }
[ -f "$SERVER_DIR/.env" ] || { echo "Missing $SERVER_DIR/.env with PEAKPLAN_TOKEN=... (the same token your app uses)."; exit 1; }

if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q .; then
  gcloud auth login
fi

# --- project + billing ---------------------------------------------------------------
if [ -f .config ]; then . ./.config; fi
if [ -z "${PROJECT:-}" ]; then
  PROJECT="peakplan-$(date +%s | tail -c 7)"
  echo "==> Creating project $PROJECT"
  gcloud projects create "$PROJECT" --name="PeakPlan"
  printf 'PROJECT=%s\nZONE=%s\nVM=%s\n' "$PROJECT" "$ZONE" "$VM" > .config
fi
BILLING=$(gcloud billing accounts list --filter=open=true --format='value(name)' | head -1)
[ -n "$BILLING" ] || { echo "No billing account. Sign up at https://console.cloud.google.com/freetrial first."; exit 1; }
gcloud billing projects link "$PROJECT" --billing-account="$BILLING" >/dev/null
echo "==> Enabling Compute Engine (takes ~1 min)"
gcloud services enable compute.googleapis.com --project "$PROJECT"

# US$1 budget alert (e-mail when you reach 50 % and 100 %). Alerts only, Google has no hard cap.
gcloud services enable billingbudgets.googleapis.com --project "$PROJECT" >/dev/null 2>&1 || true
if ! gcloud billing budgets list --billing-account="$BILLING" --format='value(displayName)' 2>/dev/null | grep -q "^PeakPlan$"; then
  gcloud billing budgets create --billing-account="$BILLING" --display-name="PeakPlan" \
    --budget-amount=1USD --threshold-rule=percent=0.5 --threshold-rule=percent=1.0 >/dev/null 2>&1 \
    && echo "==> Budget alert at US\$1 created" \
    || echo "!! Could not create the budget alert automatically; add one in Billing → Budgets & alerts."
fi

# --- VM ----------------------------------------------------------------------------
if ! gcloud compute instances describe "$VM" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
  echo "==> Creating free-tier VM"
  gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" \
    --machine-type=e2-micro \
    --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=30GB --boot-disk-type=pd-standard \
    --no-service-account --no-scopes \
    --labels=app=peakplan
fi

echo "==> Waiting for SSH"
for i in $(seq 1 20); do
  gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --quiet --command "true" >/dev/null 2>&1 && break
  sleep 10
done

PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM" WITH_DATA="$WITH_DATA" ./push.sh

cat <<MSG

✅ Done. Your PeakPlan server runs 24/7 on Google Cloud.
   Next: put the https://peakplan.<…>.ts.net address shown above into the iPhone app
   (Settings → Server URL → Connect) and update the Claude connector.
MSG
