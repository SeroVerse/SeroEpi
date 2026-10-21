#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Fetch the latest organisation-wide announcement from SeroVerse/.github Discussions.

Queries the GitHub GraphQL API for the newest discussion in the `Announcements`
category of `SeroVerse/.github` and writes `docs/overrides/announcement.html`
so Zensical's `{% block announce %}` in `docs/overrides/main.html` renders a
dismissible banner across all SeroVerse documentation sites.
"""

from __future__ import annotations

import html
import json
import os
import subprocess
import urllib.request
from pathlib import Path

REPO_OWNER = "SeroVerse"
REPO_NAME = ".github"
CATEGORY_NAME = "Announcements"
ANNOUNCEMENT_FILE = Path("docs/overrides/announcement.html")


def _resolve_github_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token.strip()
    try:
        proc = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return proc.stdout.strip() or None
    except Exception:
        return None


def main() -> None:
    query = f"""
    query {{
      repository(owner: "{REPO_OWNER}", name: "{REPO_NAME}") {{
        discussions(first: 20, orderBy: {{field: CREATED_AT, direction: DESC}}) {{
          nodes {{
            title
            url
            category {{
              name
            }}
          }}
        }}
      }}
    }}
    """

    token = _resolve_github_token()
    if not token:
        print("No GITHUB_TOKEN found; skipping announcement fetch.")
        return

    req = urllib.request.Request("https://api.github.com/graphql", method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")

    try:
        payload = json.dumps({"query": query}).encode("utf-8")
        with urllib.request.urlopen(req, data=payload, timeout=10) as response:
            data = json.loads(response.read())

        discussions = data.get("data", {}).get("repository", {}).get("discussions", {}).get("nodes", [])
        announcement = next((d for d in discussions if d.get("category", {}).get("name") == CATEGORY_NAME), None)

        if announcement:
            title = html.escape(announcement["title"])
            url = html.escape(announcement["url"], quote=True)
            banner_html = f'<strong>📢 Latest Update:</strong> <a href="{url}">{title}</a>'
            ANNOUNCEMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
            ANNOUNCEMENT_FILE.write_text(banner_html, encoding="utf-8")
            print(f"Wrote announcement banner: {announcement['title']}")
        elif ANNOUNCEMENT_FILE.exists():
            ANNOUNCEMENT_FILE.unlink()
            print("No active announcement found; removed existing announcement.html.")
        else:
            print("No active announcement found in SeroVerse/.github.")

    except Exception as exc:
        print(f"Failed to fetch announcement: {exc}")


if __name__ == "__main__":
    main()
