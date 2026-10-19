"""
plot_adf_graph.py  —  draw the compiled ADF graph (Graphviz).

Reads the graph the AIE compiler actually built, Work/temp/project.json
(kernels, PLIOs, packet merges and every connection with its port type),
writes it as Graphviz DOT and renders PNG + SVG with `dot`:

    report_figures/adf_graph_<Variant>.dot / .png / .svg

Edge styles: memory buffer (thick blue), stream (black), packet stream
(dashed orange). Kernels are labelled with their class, instance name
and source file; a broadcast stream is drawn as one edge per receiver.
With --collapse N, runs of more than N identical kernels (same class, same
connections) are drawn as the first N - 1, a "..." node and the last one.

Only the Python standard library and the Graphviz `dot` program are needed;
the script is the same in ACO_Aiengine, GA_AIE and PSO_AI_Engine.

Usage (from the project folder):
    python3 scripts/plot_adf_graph.py --work build/x86sim_FPED/Work --name FPED
"""
import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

EDGE_STYLE = {
    "mem":       'color="#1f5fbf", penwidth=2.2',
    "stream":    'color="#202020", penwidth=1.1',
    "pktstream": 'color="#d9822b", penwidth=1.3, style=dashed',
}
EDGE_LABEL = {"mem": "buffer", "stream": "stream", "pktstream": "pkt stream"}


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True, help="aiecompiler Work directory")
    ap.add_argument("--name", required=True, help="label used in the title and file names")
    ap.add_argument("--out-dir", type=Path, default=PROJECT / "report_figures")
    ap.add_argument("--collapse", type=int, default=8, help="max identical kernels drawn per group (0 = all)")
    args = ap.parse_args()

    g = json.loads((args.work / "temp" / "project.json").read_text())
    types, insts, ports = g["blockTypes"], g["blockInstances"], g["portInstances"]

    def short(inst):
        q = insts[inst]["qualifiedName"]
        return q.split(".", 1)[1] if "." in q else q

    def port_type(inst, port_key):
        port = port_key.split("_", 1)[1]
        return types[insts[inst]["blockType"]]["ports"][port]["type"]

    # edges with the type of the source port (destination for PLIO sources)
    edges = []
    for net in g["nets"].values():
        s, d = net["srcInstance"], net["dstInstance"]
        t = port_type(s, net["srcPort"])
        if types[insts[s]["blockType"]]["type"] == "plio":
            t = port_type(d, net["dstPort"])
        edges.append((s, d, t, ports[net["srcPort"]]["portName"], ports[net["dstPort"]]["portName"]))

    # collapse long runs of identical kernels (e.g. 30 workers)
    hidden, ellipsis_nodes = set(), []
    if args.collapse > 0:
        sig = defaultdict(list)
        for i, inst in insts.items():
            bt = types[inst["blockType"]]
            if bt["type"] != "function":
                continue
            ins = tuple(sorted(s for s, d, *_ in edges if d == i))
            outs = tuple(sorted(d for s, d, *_ in edges if s == i and types[insts[d]["blockType"]]["type"] != "pktmerge"))
            merge = tuple(sorted(d for s, d, *_ in edges if s == i and types[insts[d]["blockType"]]["type"] == "pktmerge"))
            sig[(bt.get("class"), ins, outs, merge)].append(i)
        for key, members in sig.items():
            if len(members) > args.collapse:
                members.sort(key=lambda m: natural_key(short(m)))
                keep = members[:args.collapse - 1] + [members[-1]]
                gone = [m for m in members if m not in keep]
                hidden.update(gone)
                ellipsis_nodes.append((key, members, gone))

    lines = [
        "digraph adf {",
        '  graph [rankdir=LR, fontname="Helvetica", fontsize=16, labelloc=t, nodesep=0.12, ranksep=1.1,',
        f'         label="ADF graph — {args.name}  ({sum(1 for v in insts.values() if types[v["blockType"]]["type"] == "function")} kernels, compiled by aiecompiler)"];',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9, arrowsize=0.7];',
    ]
    for i, inst in sorted(insts.items(), key=lambda kv: natural_key(short(kv[0]))):
        if i in hidden:
            continue
        bt = types[inst["blockType"]]
        if bt["type"] == "plio":
            direction = "input" if any(p["direction"] == "out" for p in bt["ports"].values()) else "output"
            lines.append(f'  {i} [shape=cds, style=filled, fillcolor="#e6e6e6", '
                         f'label="{direction} PLIO\\n{inst.get("logicalName", short(i))}\\n{inst.get("dataFile", "")}"];')
        elif bt["type"] == "pktmerge":
            lines.append(f'  {i} [shape=trapezium, orientation=270, style=filled, fillcolor="#fde9d4", '
                         f'label="pktmerge<{sum(1 for p in bt["ports"] if p.startswith("pi"))}>\\n{short(i)}"];')
        elif bt["type"] == "function":
            colour = "#cfe2ff" if "aster" in bt.get("class", "") else "#d8f0d2"
            lines.append(f'  {i} [shape=box, style="rounded,filled", fillcolor="{colour}", '
                         f'label="{bt.get("class", "kernel")}\\n{short(i)}\\n{bt.get("srcFile", "")}"];')
        else:
            lines.append(f'  {i} [shape=ellipse, label="{bt["type"]}\\n{short(i)}"];')
    for n, (key, members, gone) in enumerate(ellipsis_nodes):
        lines.append(f'  dots{n} [shape=box, style="rounded,dashed", label="... {len(gone)} more {key[0]}\\n'
                     f'({short(gone[0])} .. {short(gone[-1])})\\nsame connections"];')

    seen = set()
    for s, d, t, sp, dp in edges:
        s2 = s if s not in hidden else next(f"dots{n}" for n, e in enumerate(ellipsis_nodes) if s in e[2])
        d2 = d if d not in hidden else next(f"dots{n}" for n, e in enumerate(ellipsis_nodes) if d in e[2])
        key = (s2, d2, t)
        if key in seen:
            continue
        seen.add(key)
        label = EDGE_LABEL.get(t, t)
        lines.append(f'  {s2} -> {d2} [{EDGE_STYLE.get(t, "")}, label="{label}"];')
    lines.append('  subgraph cluster_legend { label="edges"; fontsize=10; style=dashed;')
    lines.append('    l1 [shape=point]; l2 [shape=point]; l3 [shape=point]; l4 [shape=point]; l5 [shape=point]; l6 [shape=point];')
    lines.append(f'    l1 -> l2 [{EDGE_STYLE["mem"]}, label="memory buffer"];')
    lines.append(f'    l3 -> l4 [{EDGE_STYLE["stream"]}, label="AXI stream (broadcast)"];')
    lines.append(f'    l5 -> l6 [{EDGE_STYLE["pktstream"]}, label="packet stream"]; }}')
    lines.append("}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"adf_graph_{args.name}"
    stem.with_suffix(".dot").write_text("\n".join(lines) + "\n")
    for fmt in ("png", "svg"):
        subprocess.run(["dot", f"-T{fmt}", "-Gdpi=130" if fmt == "png" else "-Gdpi=72",
                        str(stem.with_suffix(".dot")), "-o", str(stem.with_suffix("." + fmt))], check=True)
    print(f"ADF graph -> {stem}.png / .svg / .dot")


if __name__ == "__main__":
    main()
