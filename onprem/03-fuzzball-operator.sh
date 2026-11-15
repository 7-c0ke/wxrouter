#!/usr/bin/env bash
# Step 3, on the CONTROL PLANE: install the Fuzzball operator from CIQ Depot and create Orchestrate.
#   source onprem/env && bash onprem/03-fuzzball-operator.sh
#
# The Depot access key is read at the prompt (not echoed), used for `helm registry login`, the operator's
# image pull secret and the FuzzballOrchestrate spec, and never written to a file by this script.
# Commands follow CIQ's cag-dct-deployment page; `microk8s helm3` / `microk8s kubectl` replace helm / kubectl.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${DEPOT_USER:?}" "${FUZZBALL_VERSION:?}" "${DOMAIN:?}" "${METALLB_IP:?}" "${REALM_ID:?}" "${OWNER_EMAIL:?}" \
  "${STORAGE_CLASS:?}" "${NFS_SERVER_IP:?}" "${NFS_PATH:?}"
command -v envsubst >/dev/null || { echo "needs envsubst (apt install gettext-base)"; exit 1; }
HELM="microk8s helm3"; K="microk8s kubectl"
CHART=oci://depot.ciq.com/fuzzball/fuzzball-images/helm/fuzzball-operator

read -r -s -p "CIQ Depot access key for ${DEPOT_USER}: " DEPOT_KEY; echo
export DEPOT_KEY

printf '%s' "$DEPOT_KEY" | $HELM registry login depot.ciq.com --username "$DEPOT_USER" --password-stdin

$HELM upgrade --install fuzzball-operator "$CHART" \
    --namespace fuzzball-system --create-namespace \
    --version "$FUZZBALL_VERSION" \
    --set "image.tag=${FUZZBALL_VERSION}" \
    --set "imagePullSecrets.name=repository-ciq-com" \
    --set "imagePullSecrets.inline.registry=depot.ciq.com" \
    --set "imagePullSecrets.inline.username=${DEPOT_USER}" \
    --set "imagePullSecrets.inline.password=${DEPOT_KEY}" \
    --set "storageClassName=${STORAGE_CLASS}"

$K -n fuzzball-system rollout status deploy -l app.kubernetes.io/name=fuzzball-operator --timeout=300s || \
    $K -n fuzzball-system get pods

render() { envsubst < "$here/fuzzball.yaml.tmpl"; }

echo "Checking fuzzball.yaml against the operator's schema (server-side dry run) ..."
render | $K apply --dry-run=server -f -
render | $K apply -f -
unset DEPOT_KEY

echo "Applied. Follow the operator with:"
echo "  $K logs -l app.kubernetes.io/name=fuzzball-operator -n fuzzball-system -f --tail=-1"
echo "When every pod in namespace fuzzball runs:  $K get pods -n fuzzball"
echo "Then the first login (README step 4)."
