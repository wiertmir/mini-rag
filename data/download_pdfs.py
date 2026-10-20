#!/usr/bin/env python3
"""Download corpus from the official IPCC site. Python standard library only."""
import csv, pathlib, time, urllib.request
BASE = pathlib.Path(__file__).resolve().parent
OUT = BASE / "pdfs"
OUT.mkdir(exist_ok=True)
with (BASE / "documents.csv").open(encoding="utf-8", newline="") as f:
    docs = list(csv.DictReader(f))
for doc in docs:
    dest = OUT / (doc["id"] + ".pdf")
    if dest.exists() and dest.read_bytes()[:4] == b"%PDF":
        print("SKIP", dest.name)
        continue
    print("GET ", doc["id"], doc["url"])
    try:
        req = urllib.request.Request(doc["url"],headers={"User-Agent":"Mozilla/5.0 RAG-Benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read()
        if not data.startswith(b"%PDF"):
            raise ValueError("Response is not a PDF")
        dest.write_bytes(data)
        print(" OK ", len(data), "bytes")
    except Exception as exc:
        print("FAIL", doc["id"], repr(exc), "— see https://www.ipcc.ch/report/ar6/ for manual download")
    time.sleep(0.5)
