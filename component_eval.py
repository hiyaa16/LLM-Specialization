from tavily import TavilyClient
from eval import evaluate
from urllib.parse import urlparse

client = TavilyClient(
    api_key="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"
)

configs = [

    {
        "name": "basic_5",
        "depth": "basic",
        "results": 5
    },

    {
        "name": "advanced_5",
        "depth": "advanced",
        "results": 5
    },

    {
        "name": "advanced_10",
        "depth": "advanced",
        "results": 10
    }

]

query = """
Recent developments in black hole science
"""

for c in configs:

    print("\n====================")
    print("CONFIG:", c["name"])
    print("====================")

    response = client.search(

        query=query,

        search_depth=c["depth"],

        max_results=c["results"]

    )

    results = response["results"]

    print("\nFOUND DOMAINS:\n")

    for r in results:

        domain = urlparse(
            r["url"]
        ).netloc

        print(domain)

    passed, score = evaluate(
        results
    )

    print("\nPASS:", passed)

    print(
        "SCORE:",
        round(score, 2)
    )

    print("\nURLS:\n")

    for r in results:

        print(
            r["url"]
        )