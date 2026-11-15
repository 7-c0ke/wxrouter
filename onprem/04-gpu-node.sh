#!/usr/bin/env bash
# Step 5, on EACH GPU SERVER (Traderui, neal-master): make it a Fuzzball compute node with its RTX 3090.
#   sudo bash onprem/04-gpu-node.sh <NFS_SERVER_IP> <DEB_DIR> <CP_IP>
# DEB_DIR holds the three .deb files for the cluster's Fuzzball version, downloaded from CIQ Depot:
#   fuzzball-substrate-<v>.linux-amd64.deb  fuzzball-substrate-orchestrate_<v>_amd64.deb
#   fuzzball-substrate-nvidia-gpu-plugin-<v>.amd64.deb
#
# Follows CIQ's cag-dct-substrate (Kubernetes tab, written for Rocky with dnf/firewalld) and external-static-nodes
# (.deb names). Ubuntu equivalents are ours: CHECK with CIQ for the supported Ubuntu versions.
set -euo pipefail
NFS_SERVER_IP=${1:?NFS server IP}; DEB_DIR=${2:?folder with the substrate .deb files}; CP_IP=${3:?control plane IP}
NFS_PATH=${NFS_PATH:-/srv/fuzzball/shared}

# 0. The GPU and its driver must already work: CIQ requires the vendor driver loaded and nvidia-smi working.
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
    echo "nvidia-smi failed: install and load the NVIDIA driver first"; exit 1; }
. /etc/os-release; echo "OS: $PRETTY_NAME"
[ "$VERSION_ID" = "23.04" ] && echo "WARNING: Ubuntu 23.04 is end of life; apt may fail. Upgrade to 24.04 first."

# 1. The shared folder from the control plane, mounted at /fuzzball/shared and kept mounted.
apt-get update -qq && apt-get install -y -qq nfs-common
mkdir -p /fuzzball/shared
cat > /etc/systemd/system/fuzzball-shared.mount <<EOF
[Unit]
Description=Fuzzball shared folder (NFS from the control plane)
After=network-online.target
Wants=network-online.target

[Mount]
What=${NFS_SERVER_IP}:${NFS_PATH}
Where=/fuzzball/shared
Type=nfs
Options=rw,hard,_netdev

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now fuzzball-shared.mount
mountpoint -q /fuzzball/shared && echo "NFS mounted at /fuzzball/shared"

# 2. Substrate, the orchestrate extension (same version as the control plane) and the NVIDIA GPU plugin.
shopt -s nullglob
debs=("$DEB_DIR"/fuzzball-substrate*.deb)
[ ${#debs[@]} -ge 2 ] || { echo "no substrate .deb files in $DEB_DIR"; exit 1; }
apt-get install -y "${debs[@]}"

# 3. Where substrate reads its config: the shared folder (CIQ, Kubernetes tab).
cat > /etc/default/fuzzball-substrate.env <<'EOF'
FB_POLL_CONFIG=/fuzzball/shared/substrate/substrate-config.yaml
FB_MOUNTNS=true
EOF

# CIQ's compose page removes host-rootfs-path for GPU nodes (that path exists only in compose containers).
# CHECK whether it applies here; this only acts if the line is present.
if [ -f /etc/fuzzball-substrate/fuzzball-substrate.yml ] && grep -q 'host-rootfs-path:' /etc/fuzzball-substrate/fuzzball-substrate.yml; then
    sed -i.bak '/host-rootfs-path:/d' /etc/fuzzball-substrate/fuzzball-substrate.yml
    echo "removed host-rootfs-path (backup: fuzzball-substrate.yml.bak)"
fi

# 4. Firewall: the control plane calls the node on 7331-7333 (CIQ). Only if ufw is active.
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
    ufw allow from "$CP_IP" to any port 7331:7333 proto tcp
fi

# Docker on a compute node can interfere with substrate's networking (CIQ external-static page).
systemctl is-active --quiet docker && echo "NOTE: Docker is running here; CIQ warns it can interfere with networking."

systemctl enable --now fuzzball-substrate.service
sleep 3
systemctl --no-pager status fuzzball-substrate.service | head -15
echo "Step 5 done on $(hostname). Follow: journalctl -u fuzzball-substrate -f"
echo "Then on the control plane: add the static provisioner (README step 6) and check: fuzzball node list"
