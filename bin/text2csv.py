#!/usr/bin/env python3

import argparse
import csv
import sys


def parse_blocks(lines):
    current = {}

    for line in lines:
        line = line.rstrip("\n")

        if line == "--":
            if current:
                yield current
                current = {}
            continue

        if line.startswith("S: "):
            current["source"] = line[3:].strip()
        elif line.startswith("T: "):
            current["target"] = line[3:].strip()
        elif line.startswith("-: ") or line.startswith("+: "):
            current["prediction"] = line[3:].strip()

    if current:
        yield current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="-")
    parser.add_argument("-o", "--output", default="-")
    args = parser.parse_args()

    infile = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    outfile = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8", newline="")

    with infile, outfile:
        writer = csv.DictWriter(outfile, fieldnames=["source", "target", "prediction"])
        writer.writeheader()

        for row in parse_blocks(infile):
            writer.writerow({
                "source": row.get("source", ""),
                "target": row.get("target", ""),
                "prediction": row.get("prediction", ""),
            })


if __name__ == "__main__":
    main()