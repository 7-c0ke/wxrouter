#!/usr/bin/env bash
# Step 1, on the CONTROL PLANE (40-core server): MicroK8s ready for the Fuzzball operator.
#   sudo -E bash onprem/01-microk8s-setup.sh      (after `source onprem/env`)
#
# CIQ's guide uses RKE2 + MetalLB + a StorageClass + NFS client tools; this is the MicroK8s equivalent (our
# adaptation, untested by CIQ). Fuzzball brings its own ingress (Kong), cert-manager and trust-manager, so the
# MicroK8s addons for those stay OFF.
set -euo pipefail
: "${METALLB_RANGE:?set METALLB_RANGE in onprem/env}" "${STORAGE_CLASS:?}"

if ! command -v microk8s >/dev/null; then
    snap install microk8s --classic
fi
microk8s status --wait-ready

# Ingress and cert-manager come from Fuzzball itself: two of each would fight over ports and CRDs.
for addon in ingress cert-manager; do
    if microk8s status --addon "$addon" 2>/dev/null | grep -q enabled; then
        echo "disabling MicroK8s addon $addon (Fuzzball deploys its own)"
        microk8s disable "$addon"
    fi
done

microk8s enable dns
microk8s enable hostpath-storage          # StorageClass microk8s-hostpath (single server)
microk8s enable helm3
microk8s enable "metallb:${METALLB_RANGE}"

# NFS client tools on every Kubernetes node (CIQ: nfs-utils / nfs-common).
apt-get update -qq && apt-get install -y -qq nfs-common

microk8s kubectl get storageclass
microk8s kubectl -n metallb-system get ipaddresspools.metallb.io 2>/dev/null || true
echo "Step 1 done. Next: onprem/02-nfs-server.sh"
