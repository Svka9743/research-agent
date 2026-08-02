import arxiv
import requests
import re
import time
from pathlib import Path

DOWNLOAD_DIR = Path("documents")
DOWNLOAD_DIR.mkdir(exist_ok=True)

SEARCH_TOPICS = [
    ("large language models", 10),
    ("retrieval augmented generation", 10),
    ("transformers", 10),
    ("prompt engineering", 5),
    ("AI agents", 5),
    ("machine learning", 10),
    ("natural language processing", 10),
    ("computer vision", 10),
    ("reinforcement learning", 10),
    ("federated learning", 5),
    ("diffusion models", 5),
    ("vector databases", 5),
    ("knowledge graphs", 5),
]


def clean_filename(name):
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_")
    return name[:150]


client = arxiv.Client()

downloaded = 0

for topic, limit in SEARCH_TOPICS:

    print(f"\nSearching: {topic}")

    search = arxiv.Search(
        query=topic,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance
    )

    for paper in client.results(search):

        filename = clean_filename(paper.title) + ".pdf"

        filepath = DOWNLOAD_DIR / filename

        if filepath.exists():
            print(f"Skipping {filename}")
            continue

        pdf_url = paper.pdf_url

        print(f"Downloading {filename}")

        try:

            response = requests.get(pdf_url, timeout=60)

            response.raise_for_status()

            with open(filepath, "wb") as f:

                f.write(response.content)

            downloaded += 1

            time.sleep(2)

        except Exception as e:

            print(e)

print(f"\nDownloaded {downloaded} papers")