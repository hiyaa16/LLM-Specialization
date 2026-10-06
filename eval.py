from urllib.parse import urlparse

gold_domains = [
   
    "nasa.gov",
    "nature.com",
    "arxiv.org",
    "mit.edu",
    "harvard.edu",
    "space.com",
    "sciencedaily.com"
]


def evaluate(results):

    trusted = 0

    for item in results:

        domain = urlparse(
            item["url"]
        ).netloc

        if any(
            g in domain
            for g in gold_domains
        ):
            trusted += 1

    total = len(results)

    score = trusted / total if total else 0

    passed = score >= 0.5

    return passed, score