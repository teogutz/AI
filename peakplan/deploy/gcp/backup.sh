#!/usr/bin/env bash
# Downloads a copy of the server database to deploy/gcp/backups/ on your Mac.
set -euo pipefail
cd "$(dirname "$0")"
. ./.config
mkdir -p backups
NAME="peakplan-$(date +%Y%m%d-%H%M).db"
gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --quiet \
  --command "sudo sqlite3 /var/lib/peakplan/peakplan.db \".backup '/tmp/$NAME'\" && sudo chmod 644 /tmp/$NAME"
gcloud compute scp "$VM:/tmp/$NAME" "backups/$NAME" --zone "$ZONE" --project "$PROJECT" --quiet
echo "Saved backups/$NAME"
