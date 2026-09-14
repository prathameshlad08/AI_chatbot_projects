import urllib.request
import xml.etree.ElementTree as ET
import json
import time
import ssl
from pathlib import Path

CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]
MAX_RESULTS_PER_CATEGORY = 500
OUTPUT_FILE = Path("data/arxiv_papers.json")

NS = {"atom": "http://www.w3.org/2005/Atom"}

SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def fetch_category(category, max_results=500):
    papers = []
    batch_size = 100
    for start in range(0, max_results, batch_size):
        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=cat:{category}&start={start}&max_results={batch_size}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        with urllib.request.urlopen(url, context=SSL_CONTEXT) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        entries = root.findall("atom:entry", NS)

        for entry in entries:
            title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
            abstract = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
            arxiv_id = entry.find("atom:id", NS).text.strip()
            published = entry.find("atom:published", NS).text.strip()

            papers.append({
                "id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "category": category,
                "published": published
            })

        print(f"{category}: fetched {len(papers)}/{max_results}")
        time.sleep(3)

        if len(entries) < batch_size:
            break

    return papers


if __name__ == "__main__":
    all_papers = []
    for cat in CATEGORIES:
        all_papers.extend(fetch_category(cat, MAX_RESULTS_PER_CATEGORY))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2)

    print(f"\nTotal papers: {len(all_papers)}")
    print(f"Saved to {OUTPUT_FILE}")