#!/usr/bin/env bash
# Step 2, on the CONTROL PLANE: the NFS share the compute nodes mount at /fuzzball/shared.
#   sudo -E bash onprem/02-nfs-server.sh      (after `source onprem/env`)
#
# CIQ (remaining-deps): export /srv/fuzzball/shared rw,sync,no_subtree_check,no_root_squash to the compute nodes.
# The compute nodes read their substrate config from this share (FB_POLL_CONFIG).
set -euo pipefail
: "${NFS_PATH:?}" "${GPU_NODES:?set GPU_NODES in onprem/env}"

apt-get update -qq && apt-get install -y -qq nfs-kernel-server
mkdir -p "$NFS_PATH"

# One export line per GPU server (they are on different subnets), added once.
for ip in $GPU_NODES; do
    line="$NFS_PATH ${ip}(rw,sync,no_subtree_check,no_root_squash)"
    grep -qxF "$line" /etc/exports || echo "$line" >> /etc/exports
done
systemctl enable --now nfs-server
exportfs -ra
exportfs -v
echo "Step 2 done. Open TCP 2049 to the GPU servers if a firewall is on. Next: onprem/03-fuzzball-operator.sh"
