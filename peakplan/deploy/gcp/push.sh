#!/usr/bin/env bash
# Uploads the server code (and optionally your database) and (re)installs it on the VM.
# Used by create.sh and update.sh.
set -euo pipefail
cd "$(dirname "$0")"
: "${PROJECT:?}" "${ZONE:?}" "${VM:?}"
SERVER_DIR="../../server"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "==> Packaging server"
COPYFILE_DISABLE=1 tar -czf "$TMP/server.tgz" -C "$SERVER_DIR" \
  --exclude node_modules --exclude dist --exclude data --exclude .env .
FILES=("$TMP/server.tgz" "setup-vm.sh")
cp "$SERVER_DIR/.env" "$TMP/peakplan.env"; FILES+=("$TMP/peakplan.env")
if [ "${WITH_DATA:-0}" = "1" ] && [ -f "$SERVER_DIR/data/peakplan.db" ]; then
  echo "==> Including your current database"
  # Consistent copy even if the local server is running.
  sqlite3 "$SERVER_DIR/data/peakplan.db" ".backup '$TMP/peakplan.db'" 2>/dev/null || cp "$SERVER_DIR/data/peakplan.db" "$TMP/peakplan.db"
  FILES+=("$TMP/peakplan.db")
fi

gcloud compute scp "${FILES[@]}" "$VM:/tmp/" --zone "$ZONE" --project "$PROJECT" --quiet
echo "==> Installing on the VM (first time: ~5 min, Tailscale will ask you to log in)"
gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --quiet --command "sudo bash /tmp/setup-vm.sh" -- -t
