#!/usr/bin/env python3
import sys

inp, outp = sys.argv[1], sys.argv[2]

def base_id(x: str) -> str:
    x = x.strip()
    if not x:
        return ""
    # remove leading @ if present
    if x.startswith("@"):
        x = x[1:]
    # remove /1 or /2 if present
    if x.endswith("/1") or x.endswith("/2"):
        return x[:-2]
    return x

seen = set()
with open(inp) as f:
    for line in f:
        b = base_id(line)
        if b:
            seen.add(b)

with open(outp, "w") as out:
    for b in sorted(seen):
        out.write(f"{b}/1\n")
        out.write(f"{b}/2\n")