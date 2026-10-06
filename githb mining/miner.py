import os
import json
import time
import shutil
import requests

from pathlib import Path
from git import Repo
from dotenv import load_dotenv


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError(
        "GITHUB_TOKEN not found. "
        "Create a .env file containing:\n\n"
        "GITHUB_TOKEN=your_token_here"
    )


HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}


DATA_DIR = Path("data")
REPO_DIR = Path("repos")

DATA_DIR.mkdir(exist_ok=True)
REPO_DIR.mkdir(exist_ok=True)


# =========================================================
# RESEARCH SETTINGS
# =========================================================

# Pipeline validated on a 30-repository sample, then a 50-repository sanity
# check (0 errors, 0 duplicates, notebook support added afterward). Now
# scaling to the full target. TARGET_SAMPLE_SIZE is a ceiling, not a
# guarantee — actual count depends on how many distinct repositories the
# search queries plus the relevance filter surface.
TARGET_SAMPLE_SIZE = 500

# GitHub's search API caps any single query at 1000 results (10 pages of
# 100) regardless of how many actually match. Per-query cap here is set well
# below that so a handful of queries comfortably clears TARGET_SAMPLE_SIZE
# once overlap and the relevance filter thin the raw results out.
GITHUB_SEARCH_MAX_PER_QUERY = 300
GITHUB_SEARCH_HARD_CAP = 1000
GITHUB_SEARCH_PER_PAGE = 100

# The search endpoint's authenticated rate limit is much lower than the REST
# API's general limit (~30 requests/minute). This delay keeps every page
# fetch comfortably under that regardless of how many queries/pages run.
REQUEST_DELAY_SECONDS = 2.5


SEARCH_QUERIES = [

    # Generic multi-agent searches
    '"multi-agent" LLM',
    '"multi agent" LLM',
    '"multi-agent system" LLM',
    '"agentic workflow" LLM',
    '"AI agents" LLM',

    # Framework ecosystems
    'LangGraph agents',
    'CrewAI agents',
    'AutoGen agents',
    '"OpenAI Agents" LLM',
    'CAMEL agents',
    'MetaGPT agents',
    'Agno agents',

]


# =========================================================
# GITHUB SEARCH
# =========================================================

def github_search(query, max_results=GITHUB_SEARCH_MAX_PER_QUERY):

    # GitHub's search API returns at most 100 results per page and refuses
    # to serve past 1000 results for a single query, so reaching a few
    # hundred results per query means paginating rather than raising
    # per_page alone.

    url = "https://api.github.com/search/repositories"

    results = []
    page = 1

    while (
        len(results) < max_results
        and (page - 1) * GITHUB_SEARCH_PER_PAGE < GITHUB_SEARCH_HARD_CAP
    ):

        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": GITHUB_SEARCH_PER_PAGE,
            "page": page,
        }

        response = None

        for attempt in range(3):

            try:

                response = requests.get(
                    url,
                    headers=HEADERS,
                    params=params,
                    timeout=30
                )

            except requests.RequestException as e:

                print(f"[ERROR] GitHub request failed: {e}")

                response = None

                break

            # Primary or secondary rate limiting: back off and retry rather
            # than aborting the whole query partway through.
            if response.status_code in (403, 429):

                wait = int(response.headers.get("Retry-After", 30)) * (attempt + 1)

                print(
                    f"[RATE LIMIT] {response.status_code} on page {page}; "
                    f"waiting {wait}s (attempt {attempt + 1}/3)"
                )

                time.sleep(wait)

                continue

            break

        if response is None or response.status_code in (403, 429):

            print(f"[ERROR] Giving up on page {page} of query after retries.")

            break

        if response.status_code != 200:

            print(
                f"[ERROR] GitHub API {response.status_code}: "
                f"{response.text[:300]}"
            )

            break

        items = response.json().get("items", [])

        if not items:
            break

        results.extend(items)

        if len(items) < GITHUB_SEARCH_PER_PAGE:
            break

        page += 1

        time.sleep(REQUEST_DELAY_SECONDS)

    return results[:max_results]


# =========================================================
# COLLECT REPOSITORIES
# =========================================================

def collect_repositories():

    repositories = {}

    for query in SEARCH_QUERIES:

        print(f"\nSearching: {query}")

        results = github_search(query)

        for repo in results:

            full_name = repo["full_name"]

            # Deduplication by owner/repository
            repositories[full_name] = {

                "name": full_name,

                "url": repo["html_url"],

                "clone_url": repo["clone_url"],

                "description": repo.get(
                    "description"
                ) or "",

                "stars": repo.get(
                    "stargazers_count",
                    0
                ),

                "forks": repo.get(
                    "forks_count",
                    0
                ),

                "language": repo.get(
                    "language"
                ),

                "topics": repo.get(
                    "topics",
                    []
                ),

                "created_at": repo.get(
                    "created_at"
                ),

                "updated_at": repo.get(
                    "updated_at"
                ),

                "license": (
                    repo.get("license", {}).get("spdx_id")
                    if repo.get("license")
                    else None
                ),

            }

        # Small delay between queries (github_search already paces its own
        # page requests internally).
        time.sleep(REQUEST_DELAY_SECONDS)


    repositories = list(
        repositories.values()
    )


    print(
        f"\nCollected {len(repositories)} "
        f"unique repositories."
    )


    return repositories


# =========================================================
# REPOSITORY RELEVANCE FILTER
# =========================================================

AGENT_KEYWORDS = [

    "agent",
    "agents",
    "multi-agent",
    "multi agent",
    "agentic",

    "autogen",
    "crewai",
    "langgraph",
    "metagpt",
    "camel",
    "agno",

    "openai agents",
    "llm agent",
    "ai agent"

]


def is_potential_agent_repo(repo):

    text = " ".join([

        repo["name"],

        repo["description"],

        " ".join(repo["topics"])

    ]).lower()


    return any(
        keyword in text
        for keyword in AGENT_KEYWORDS
    )


# =========================================================
# CLONE REPOSITORY
# =========================================================

def clone_repository(repo):

    repo_name = repo["name"].replace(
        "/",
        "__"
    )

    destination = (
        REPO_DIR /
        repo_name
    )


    # Already successfully cloned
    if destination.exists():

        print(
            f"[SKIP] Already exists: "
            f"{repo['name']}"
        )

        return destination


    print(
        f"[CLONE] {repo['name']}"
    )


    try:

        Repo.clone_from(

            repo["clone_url"],

            destination,

            depth=1

        )

        print(
            f"[OK] {repo['name']}"
        )

        return destination


    except Exception as e:

        print(
            f"[ERROR] Could not clone "
            f"{repo['name']}: {e}"
        )


        # Remove incomplete clone
        if destination.exists():

            try:

                shutil.rmtree(
                    destination
                )

                print(
                    f"[CLEANUP] Removed "
                    f"incomplete clone: "
                    f"{repo['name']}"
                )

            except Exception as cleanup_error:

                print(
                    f"[WARNING] Could not "
                    f"remove incomplete clone: "
                    f"{cleanup_error}"
                )


        return None


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # STEP 1: Search GitHub
    # -----------------------------------------------------

    repositories = (
        collect_repositories()
    )


    # -----------------------------------------------------
    # STEP 2: Filter candidate repositories
    # -----------------------------------------------------

    filtered = [

        repo

        for repo in repositories

        if is_potential_agent_repo(repo)

    ]


    print(
        f"\nPotential agent repositories: "
        f"{len(filtered)}"
    )


    # -----------------------------------------------------
    # STEP 3: Limit to target sample size
    # -----------------------------------------------------

    filtered = filtered[
        :TARGET_SAMPLE_SIZE
    ]


    print(
        f"Mining sample: "
        f"{len(filtered)} repositories"
    )


    # -----------------------------------------------------
    # STEP 4: Save metadata
    # -----------------------------------------------------

    output_file = (
        DATA_DIR /
        "repositories.json"
    )


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            filtered,

            f,

            indent=2,

            ensure_ascii=False

        )


    print(
        f"\nSaved metadata to "
        f"{output_file}"
    )


    # -----------------------------------------------------
    # STEP 5: Clone repositories
    # -----------------------------------------------------

    successful = 0
    failed = 0


    for repo in filtered:

        result = clone_repository(
            repo
        )


        if result:

            successful += 1

        else:

            failed += 1


        # Avoid aggressive requests
        time.sleep(0.5)


    # -----------------------------------------------------
    # STEP 6: Summary
    # -----------------------------------------------------

    print("\n" + "=" * 50)

    print("MINING SUMMARY")

    print("=" * 50)

    print(
        f"Candidate repositories : "
        f"{len(filtered)}"
    )

    print(
        f"Successfully cloned     : "
        f"{successful}"
    )

    print(
        f"Failed                   : "
        f"{failed}"
    )

    print("=" * 50)


if __name__ == "__main__":

    main()