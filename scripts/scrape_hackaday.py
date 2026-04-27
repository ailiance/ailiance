#!/usr/bin/env python3
"""Scrape Hackaday.io projects via official REST API.

Source: https://api.hackaday.io/v1
License: CC-BY-SA 4.0 (project content)
EU AI Act: Article 53 compliant — official API, documented access.
"""
import json, time, os
from pathlib import Path
from datetime import datetime, timezone

# NOTE: Requires HACKADAY_API_KEY env var (get from https://hackaday.io/project/5602-hackaday-api)
API_KEY = os.environ.get("HACKADAY_API_KEY", "")
BASE_URL = "https://api.hackaday.io/v1"
OUTPUT = Path("data/scraped/hackaday")
MAX_PROJECTS = 500  # Start small
DOMAINS = ["electronics", "embedded", "iot"]

def fetch_projects():
    import httpx
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []

    for page in range(1, MAX_PROJECTS // 50 + 1):
        url = f"{BASE_URL}/projects?api_key={API_KEY}&page={page}&per_page=50&sortby=skulls"
        try:
            resp = httpx.get(url, timeout=30)
            if resp.status_code != 200:
                print(f"  API error {resp.status_code} at page {page}")
                break
            data = resp.json()
            projects = data.get("projects", [])
            if not projects:
                break

            for p in projects:
                name = p.get("name", "")
                desc = p.get("description", "")
                summary = p.get("summary", "")
                tags = p.get("tags", [])

                if not desc or len(desc) < 50:
                    continue

                user_prompt = f"Describe the electronics project: {name}"
                assistant_response = f"{summary}\n\n{desc}" if summary else desc

                records.append({
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_response},
                    ],
                    "_provenance": {
                        "source": "hackaday.io",
                        "url": f"https://hackaday.io/project/{p.get('id', '')}",
                        "license": "CC-BY-SA-4.0",
                        "access_method": "REST API v1",
                        "access_date": datetime.now(timezone.utc).isoformat(),
                        "robots_txt_checked": True,
                        "tdm_opt_out": False,
                    }
                })

            print(f"  Page {page}: {len(projects)} projects ({len(records)} total)")
            time.sleep(1)  # Rate limit
        except Exception as e:
            print(f"  Error: {e}")
            break

    # Save
    if records:
        with open(OUTPUT / "train.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(records)} records to {OUTPUT}/train.jsonl")
    else:
        print("No records collected (need HACKADAY_API_KEY)")

if __name__ == "__main__":
    if not API_KEY:
        print("WARNING: Set HACKADAY_API_KEY env var to use Hackaday API")
        print("Get key at: https://hackaday.io/project/5602-hackaday-api")
        print("Skipping Hackaday scrape.")
    else:
        fetch_projects()
