# Run the PeakPlan server on Google Cloud (Always Free)

This guide sets the server up to run 24/7 within Google's free tier:

* **Machine:** one e2-micro VM in `us-central1`.
* **Disk:** a 30 GB *standard* disk.
* **Backups:** daily, kept on the VM's own disk. Snapshots would cost money, so none are used.
* **Address:** a fixed HTTPS address through Tailscale Funnel. There are no open ports and no domain to buy.

## What you need

* **Google Cloud account:** sign up for the free trial at https://console.cloud.google.com/freetrial. Use a low-limit virtual debit card.
* **Google Cloud CLI:** `brew install --cask google-cloud-sdk`
* **The server token** in `server/.env` (`PEAKPLAN_TOKEN=...`), the same token your app uses.

## Commands

```bash
cd peakplan/deploy/gcp
./create.sh --with-data   # first time: creates everything and copies your current data
./update.sh               # after `git pull`: deploys new server code, keeps data and token
./backup.sh               # downloads a copy of the database to deploy/gcp/backups/
```

## Costs to watch

* **The 90-day trial:** Google can't charge you during the 90-day trial unless you upgrade the account. After that you must upgrade to keep the VM running.
* **Budget alert:** `create.sh` adds a **US$1 budget alert**. It e-mails you; it doesn't stop anything.
* **IPv4 address:** Google may charge a few dollars a month for the VM's public IPv4 address. A few days after creating the VM, check **Billing → Reports**.
* **Don't change these:** the region, machine type or disk type, and don't add snapshots. Those leave the free tier.
* **To stop all costs:** delete the project with `gcloud projects delete $(. ./.config; echo $PROJECT)`. Run `./backup.sh` first.
