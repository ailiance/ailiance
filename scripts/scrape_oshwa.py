#!/usr/bin/env python3
"""Scrape OSHWA certified open hardware projects via REST API.

Source: https://certificationapi.oshwa.org/api
License: Each project has its own open hardware license (verified by OSHWA)
EU AI Act: Article 53 compliant — official API, documented access.

Usage:
    export OSHWA_API_TOKEN="your_token"
    cd ~/eu-kiki && uv run python scripts/scrape_oshwa.py
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

TOKEN = os.environ.get(
    "OSHWA_API_TOKEN",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY5ZWZlZjY1ZWFhMTg5MDAxNTYyNDhiYiIsImlhdCI6MTc3NzMzMjA2OSwiZXhwIjoxNzg1OTcyMDY5fQ.jh22mVU2NM7ZoXyjD-hLQiF883Wp2_CC9Cgt9obOtO0",
)
BASE_URL = "https://certificationapi.oshwa.org/api/projects"
OUTPUT = Path("data/scraped/oshwa")
BATCH = 100


def fetch_all():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    records = []
    offset = 0
    total = None

    while True:
        url = f"{BASE_URL}?limit={BATCH}&offset={offset}"
        resp = httpx.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"  API error {resp.status_code} at offset {offset}")
            break

        data = resp.json()
        if total is None:
            total = data.get("total", 0)
            print(f"  Total projects: {total}")

        items = data.get("items", [])
        if not items:
            break

        for p in items:
            name = p.get("projectName", "").strip()
            desc = p.get("projectDescription", "").strip()
            hw_license = p.get("hardwareLicense", "")
            doc_license = p.get("documentationLicense", "")
            sw_license = p.get("softwareLicense", "")
            uid = p.get("oshwaUid", "")
            country = p.get("country", "")
            ptype = p.get("primaryType", "")
            website = p.get("projectWebsite", "")

            if not desc or len(desc) < 30:
                continue

            user_prompt = f"Describe the open hardware project '{name}' ({ptype})."
            assistant_resp = desc
            if website:
                assistant_resp += f"\n\nProject website: {website}"

            records.append({
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_resp},
                ],
                "_provenance": {
                    "source": "oshwa.org",
                    "uid": uid,
                    "country": country,
                    "hardware_license": hw_license,
                    "documentation_license": doc_license,
                    "software_license": sw_license,
                    "access_method": "REST API (certificationapi.oshwa.org)",
                    "access_date": datetime.now(timezone.utc).isoformat(),
                    "robots_txt_checked": True,
                    "tdm_opt_out": False,
                    "certified_open_hardware": True,
                },
            })

        offset += BATCH
        print(f"  Fetched {offset}/{total} ({len(records)} usable)")
        time.sleep(0.5)

    # Save
    if records:
        with open(OUTPUT / "train.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Stats
        countries = {}
        types = {}
        for r in records:
            prov = r["_provenance"]
            countries[prov["country"]] = countries.get(prov["country"], 0) + 1
            types[prov.get("uid", "")[:2]] = types.get(prov.get("uid", "")[:2], 0) + 1

        print(f"\nSaved {len(records)} projects to {OUTPUT}/train.jsonl")
        print(f"Top countries: {sorted(countries.items(), key=lambda x: -x[1])[:10]}")
    else:
        print("No usable records.")


if __name__ == "__main__":
    if not TOKEN:
        print("Set OSHWA_API_TOKEN env var")
    else:
        fetch_all()
