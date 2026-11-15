# Onprem: Fuzzball on our own servers

Run CIQ Fuzzball on our hardware: the control plane (Fuzzball Orchestrate) on the **40-core MicroK8s server**,
and the two **RTX 3090 servers** as GPU compute nodes. After that, `deploy/fb.py` deploys wxrouter onto this
cluster the same way it does on the event cluster (see [`wxrouter-on-onprem.md`](wxrouter-on-onprem.md)).

> **Source and status.** The steps follow CIQ's Fuzzball 4.4 cluster admin guide (links below). CIQ developed
> its Kubernetes path on **RKE2** and says other distributions "should also work but are untested at this
> time". **MicroK8s is not mentioned in CIQ's docs: the MicroK8s parts here are our adaptation.** Nothing in
> this folder has been run yet. Every place we had to guess is marked `CHECK`.

## Layout

| machine | role | runs |
|---|---|---|
| 40-core server (`CP_IP`) | control plane | MicroK8s, MetalLB, the Fuzzball operator → Orchestrate (API, Keycloak, Kong ingress, NATS JetStream, database, cert-manager, trust-manager), NFS server for `/srv/fuzzball/shared` |
| `Traderui` 10.0.0.134 (RTX 3090) | compute node | `fuzzball-substrate` + orchestrate extension + NVIDIA GPU plugin |
| `neal-master` 178.168.1.4 (RTX 3090) | compute node | the same |

## Before you start

| need | why |
|---|---|
| **CIQ Depot** user + access key, and the Fuzzball **version** from the CIQ portal ("my products") | the operator chart, its images and the substrate packages come from `depot.ciq.com` |
| The substrate `.deb` packages for that version (`fuzzball-substrate`, `fuzzball-substrate-orchestrate`, `fuzzball-substrate-nvidia-gpu-plugin`), downloaded from Depot onto each GPU server | CIQ documents `.deb` names but no Ubuntu apt repository (`CHECK`) |
| **Network**: both GPU servers ↔ control plane | compute → control plane: TCP **443** (API, Kong), **4222** (NATS), **5353** (DNS), **2049** (NFS); control plane → compute: TCP **7331-7333**. The two GPU servers are on different subnets (10.0.0.x, 178.168.1.x): both must reach the MetalLB IP and the NFS server |
| A free IP (or range) for **MetalLB** that both GPU servers can reach | it becomes the cluster's address; the domain is `<IP>.nip.io` |
| **DNS**: the GPU servers resolve `api.<domain>` and `nats.<domain>` | nip.io needs internet DNS; otherwise add `/etc/hosts` entries |
| **NVIDIA driver** loaded on each GPU server, `nvidia-smi` works | CIQ states no driver/CUDA version (`CHECK`) |
| Disk: control plane ≥ 100 GB free; each GPU server room for images (the PyTorch image is >10 GB) | `neal-master` has ~28 GB free: give it more or a data disk |
| `Traderui` runs **Ubuntu 23.04 (end of life)** | upgrading to 24.04 first is the safe choice; CIQ lists no Ubuntu version for compute nodes (`CHECK`) |
| Control plane size | CIQ: "at least 10 cores and 16 GBs of memory" for the server node |

Fill [`env.example`](env.example) → `env` on the control plane. **Depot credentials are never written to a file:**
the scripts ask for the access key at the prompt.

## Order

| step | where | file |
|---|---|---|
| 1. MicroK8s base: DNS, storage, Helm, MetalLB; ingress addon off | control plane | [`01-microk8s-setup.sh`](01-microk8s-setup.sh) |
| 2. NFS share `/srv/fuzzball/shared` for the compute nodes | control plane | [`02-nfs-server.sh`](02-nfs-server.sh) |
| 3. Fuzzball operator + `FuzzballOrchestrate` | control plane | [`03-fuzzball-operator.sh`](03-fuzzball-operator.sh), [`fuzzball.yaml.tmpl`](fuzzball.yaml.tmpl) |
| 4. First login, a group for the team | your laptop / control plane | below |
| 5. Each GPU server: NFS mount, substrate packages, GPU plugin, service | each GPU server | [`04-gpu-node.sh`](04-gpu-node.sh) |
| 6. Tell Fuzzball the two servers are ours: static provisioner | control plane | [`static-provisioner.yaml`](static-provisioner.yaml) |
| 7. Check: both nodes, one GPU each, a test GPU job | anywhere with the CLI | [`gpu-test.fz`](gpu-test.fz) |
| 8. wxrouter on this cluster | your laptop | [`wxrouter-on-onprem.md`](wxrouter-on-onprem.md) |

### 4. First login

The deployment prints the Keycloak URL and the admin passwords (CIQ: `deployment-info-wait.sh`; location `CHECK`).
Then, with the realm id you chose in `env`:

```bash
fuzzball context create onprem api.$DOMAIN https://auth.$DOMAIN/auth/realms/$REALM_ID fuzzball-cli
fuzzball context login --direct -u "$OWNER_EMAIL" -p '<password>'
fuzzball group create team14   # or your name; wxrouter's Fuzzfiles say `use: team14` for the volume
```

The realm URL needs `/auth`. A local admin works too: `fuzzball context create fuzzball-local api.$DOMAIN local fuzzball-cli`,
then `fuzzball context login -u fuzzball-admin@local -p '<password>'`.

### 6. Static provisioner

```bash
fuzzball cluster config get | tee cluster_config.orig.yaml
# add the definition from static-provisioner.yaml under `definitions:`, save as cluster_config.new.yaml
fuzzball cluster config set cluster_config.new.yaml
```

### 7. Check

```bash
microk8s kubectl get pods -n fuzzball
microk8s kubectl get pods -A --field-selector status.phase!=Running,status.phase!=Succeeded   # should be empty
fuzzball node list                          # Traderui and neal-master, Ready
fuzzball node list --available-resources    # nvidia.com/gpu: 1 on each
fuzzball workflow start -n gpu-test onprem/gpu-test.fz
fuzzball workflow log <id> on-traderui      # nvidia-smi output naming the RTX 3090
```

On a GPU server: `sudo journalctl -u fuzzball-substrate -f`.

## Sources (CIQ Fuzzball 4.4 docs)

- Requirements: https://docs.ciq.com/fuzzball/cluster-admin-guide/cag-dct-requirements
- Kubernetes deployment (operator, `FuzzballOrchestrate`): https://docs.ciq.com/fuzzball/cluster-admin-guide/cag-dct-deployment
- Remaining dependencies (MetalLB, storage, NFS): https://docs.ciq.com/fuzzball/appendices/k8s-cluster/remaining-deps
- Compute nodes (substrate, Kubernetes tab): https://docs.ciq.com/fuzzball/cluster-admin-guide/cag-dct-substrate
- External static nodes, `.deb` packages, static provisioner: https://docs.ciq.com/fuzzball/cloud-admin-guide/external-static-nodes
- First login: https://docs.ciq.com/fuzzball/initial-configuration/initial-login

## Not found in CIQ's docs (decide or ask CIQ)

- MicroK8s; Kubernetes version; component versions (Kong, cert-manager, trust-manager).
- An Ubuntu apt repository for the substrate packages; Ubuntu versions supported for compute nodes.
- NVIDIA driver / CUDA version requirements; RTX 3090 specifics.
- The exact `FuzzballOrchestrate` YAML for 4.4 (our template follows the documented fields; the operator's own
  schema decides: step 3 runs a server-side dry run first and stops on any unknown field).
- How Fuzzball persistent volumes (wxrouter's `wxdesk`) are backed on-prem: research before step 8.
