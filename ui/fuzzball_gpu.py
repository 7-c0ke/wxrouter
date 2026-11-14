"""The team's GPUs as Fuzzball's scheduler sees them, for the Control Room.

Every `interval_s` (15 s) the poller asks the logged-in fuzzball CLI:
    fuzzball node list -o json             every node: status, health, reliability score, team label
    fuzzball node show <id> -o json        for the team's nodes only: the GPU (model, UUID), allocated or free,
                                           and the jobs holding it (workflow, GPUs, cores and memory asked for)
    fuzzball workflow list -o json         the team's own workflows, to name the ones that hold a GPU
A workflow that holds a team GPU and is not in that list is in a group other than the CLI's selected one
(often the team's shared group): Fuzzball does not say which, so it is shown as "not in the CLI's group".

Which nodes are the team's: the ones whose `definitionId` is the group's short name (team14 for
team14@fb-hack.local, from `fuzzball context show`), plus any named with --fuzzball-node box-a=HOST, which
also ties a node to a box card. Each change between two polls is one line in the Events card (src "cluster"):
a GPU taken or freed (and by which workflow), a node's health or status changing, its reliability score
dropping or recovering.

Read-only: the poller never changes anything on the cluster. Without a CLI or a group it is off and says why.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

GPU = "nvidia.com/gpu"


def short(text: Optional[str]) -> str:
    return str(text or "").removeprefix("NODE_HEALTH_STATE_").removeprefix("NODE_STATUS_").lower()


def node_view(show: dict, ours: dict[str, str]) -> dict:
    """One node from `node show -o json`, reduced to what the page and the events need.
    `ours`: workflow id -> name for the team's own workflows."""
    res = show.get("resource") or {}
    ann = res.get("annotations") or {}
    devs = ((res.get("pluginDevices") or {}).get("list") or {}).get(GPU, {}).get("devices") or []
    summary = next((d for d in (show.get("resourceSummary") or {}).get("deviceSummaries") or []
                    if d.get("deviceType") == GPU), {})
    holders: dict[str, dict] = {}
    for j in show.get("runningJobs") or []:
        gpus = sum(int(d.get("count") or 0) for d in j.get("requestedDeviceDetails") or [] if d.get("deviceType") == GPU)
        wid = j.get("workflowId") or "?"
        h = holders.setdefault(wid, {"workflow": wid, "name": ours.get(wid), "ours": wid in ours,
                                     "gpus": 0, "cores": 0, "memory_gb": 0, "jobs": 0})
        h["gpus"] += gpus
        h["cores"] += int(j.get("requestedCores") or 0)
        h["memory_gb"] += float(j.get("requestedMemoryGb") or 0)
        h["jobs"] += 1
    gpu_holders = [h for h in holders.values() if h["gpus"] > 0]
    rs = show.get("resourceSummary") or {}
    return {
        "node": show.get("id"), "host": show.get("hostName"),
        "status": short(show.get("status")), "health": short(show.get("healthState")),
        "score": show.get("reliabilityScore"),
        "gpu": {"model": ann.get("nvidia.com/gpu.model"), "arch": ann.get("nvidia.com/gpu.arch"),
                "uuids": [d.get("id") for d in devs],
                "total": summary.get("totalCount", len(devs)), "free": summary.get("availableCount"),
                "allocated": any(d.get("allocated") for d in devs)},
        "held_by": gpu_holders,
        "cores_free": rs.get("availableCores"), "cores_total": rs.get("totalCores"),
        "memory_free_gb": rs.get("availableMemoryGb"), "memory_total_gb": rs.get("totalMemoryGb"),
    }


def holder_text(h: dict) -> str:
    return f"workflow {h['name']} ({h['workflow'][:8]})" if h.get("name") else f"workflow {h['workflow'][:8]} (not in the CLI's group)"


class FuzzballGPUs:
    def __init__(self, hub, cli: Optional[str], group: Optional[str] = None, nodes: Optional[dict[str, str]] = None,
                 interval_s: float = 15.0):
        """group: the team label as in the nodes' definitionId (team14); nodes: box -> host name."""
        self.hub, self.cli, self.group, self.interval_s = hub, cli, group, interval_s
        self.box_of = {host: box for box, host in (nodes or {}).items()}
        self.nodes: dict[str, dict] = {}        # host -> node_view
        self.ours: dict[str, str] = {}
        self.status = "off" if not cli else "starting"
        self.reason = "no fuzzball CLI on this machine" if not cli else ""
        self.polled_at: Optional[float] = None

    def info(self) -> dict:
        return {"status": self.status, "reason": self.reason, "group": self.group, "polled_at": self.polled_at,
                "nodes": [{**v, "box": self.box_of.get(v["host"])} for v in self.nodes.values()]}

    # ------------------------------------------------------------ changes -> events (pure, tested)
    def _add(self, level: str, text: str, host: str) -> None:
        extra = {"node": host}
        if self.box_of.get(host):
            extra["box"] = self.box_of[host]
        self.hub.add("cluster", level, text, **extra)

    def update(self, views: list[dict]) -> None:
        """Take the new node views; add an event for each change since the last poll."""
        for v in views:
            host, old = v["host"], self.nodes.get(v["host"])
            self.nodes[host] = v
            label = f"{host}" + (f" ({self.box_of[host]})" if self.box_of.get(host) else "")
            if old is None:
                held = ", ".join(holder_text(h) for h in v["held_by"])
                self._add("info", f"{label}: {v['gpu']['model'] or 'GPU'} {'busy, held by ' + held if v['gpu']['allocated'] else 'free'}"
                          f"; {v['health']}, score {v['score']}", host)
                continue
            was = {h["workflow"]: h for h in old["held_by"]}
            now = {h["workflow"]: h for h in v["held_by"]}
            for wid in now.keys() - was.keys():
                h = now[wid]
                self._add("info" if h["ours"] else "bad", f"{label}: GPU taken by {holder_text(h)}", host)
            for wid in was.keys() - now.keys():
                self._add("end", f"{label}: GPU released by {holder_text(was[wid])}", host)
            if old["gpu"]["allocated"] and not v["gpu"]["allocated"]:
                self._add("up", f"{label}: GPU free", host)
            if v["health"] != old["health"]:
                self._add("up" if v["health"] == "healthy" else "bad", f"{label}: health {old['health']} -> {v['health']}", host)
            if v["status"] != old["status"]:
                self._add("up" if v["status"] == "ready" else "bad", f"{label}: status {old['status']} -> {v['status']}", host)
            if isinstance(v["score"], (int, float)) and isinstance(old["score"], (int, float)) and v["score"] != old["score"]:
                self._add("bad" if v["score"] < old["score"] else "up",
                          f"{label}: reliability score {old['score']} -> {v['score']}", host)

    def gpu_of_box(self, box: str) -> Optional[dict]:
        host = next((h for h, b in self.box_of.items() if b == box), None)
        return self.nodes.get(host) if host else None

    # ------------------------------------------------------------ the CLI
    async def _json(self, *args: str, timeout: float = 30) -> object:
        p = await asyncio.create_subprocess_exec(self.cli, *args, "-o", "json",
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(p.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            p.kill()
            raise RuntimeError("fuzzball did not answer within 30 s")
        if p.returncode != 0:
            lines = (err or out).decode(errors="replace").strip().splitlines()
            raise RuntimeError(lines[-1][:200] if lines else f"fuzzball exited {p.returncode}")
        return json.loads(out or b"null")

    async def find_group(self) -> Optional[str]:
        ctx = await self._json("context", "show")
        name = (ctx or {}).get("groupName") if isinstance(ctx, dict) else None
        return name.split("@")[0] if name else None

    async def poll_once(self) -> None:
        was = self.status
        try:
            if not self.group:
                self.group = await self.find_group()
            listed = await self._json("node", "list") or []
            hosts = set(self.box_of)
            team = [n for n in listed if n.get("definitionId") == self.group or n.get("hostName") in hosts]
            try:
                self.ours = {w["id"]: w.get("name") for w in (await self._json("workflow", "list") or [])}
            except RuntimeError:
                pass    # keep the last list; holders are then named from it
            shows = await asyncio.gather(*(self._json("node", "show", n["id"]) for n in team))
            self.update([node_view(s, self.ours) for s in shows if isinstance(s, dict)])
            self.status = "ok" if team else "no nodes"
            self.reason = "" if team else f"no node is labelled {self.group!r}; name them with --fuzzball-node box-a=HOST"
            if was == "unreachable":
                self.hub.add("cluster", "up", "Fuzzball node API reachable again")
        except (RuntimeError, ValueError, OSError, KeyError) as e:
            reason = str(e)
            if "connection refused" in reason or "Unavailable" in reason:
                reason = "Fuzzball API unreachable from this machine"
            if was != "unreachable":
                self.hub.add("cluster", "bad", f"GPU view: {reason}")
            self.status, self.reason = "unreachable", reason
        self.polled_at = time.time()

    async def run(self) -> None:
        if self.status == "off":
            return
        while True:
            await self.poll_once()
            await asyncio.sleep(self.interval_s)
