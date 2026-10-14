#!/usr/bin/env bash
# Deploys the latest server code to the VM (after `git pull`). Your data and token are kept.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .config ] || { echo "Run ./create.sh first."; exit 1; }
. ./.config
PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM" WITH_DATA=0 ./push.sh
