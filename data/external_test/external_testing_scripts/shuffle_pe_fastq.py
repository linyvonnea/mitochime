#!/usr/bin/env python3
import random, sys

r1_in, r2_in, r1_out, r2_out, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])

def read_fastq(path):
    with open(path) as f:
        while True:
            h = f.readline()
            if not h:
                break
            s = f.readline(); p = f.readline(); q = f.readline()
            yield (h, s, p, q)

r1 = list(read_fastq(r1_in))
r2 = list(read_fastq(r2_in))
if len(r1) != len(r2):
    raise SystemExit("ERROR: R1 and R2 read counts differ.")

idx = list(range(len(r1)))
random.Random(seed).shuffle(idx)

with open(r1_out, "w") as o1, open(r2_out, "w") as o2:
    for i in idx:
        o1.writelines(r1[i])
        o2.writelines(r2[i])