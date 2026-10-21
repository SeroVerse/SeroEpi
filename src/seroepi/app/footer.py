"""SeroVerse-themed application footer populated dynamically from `importlib.metadata`."""

import importlib.metadata
import tomllib
from dataclasses import dataclass
from email.utils import parseaddr
from functools import lru_cache
from pathlib import Path

from htmltools import HTML, Tag
from shiny import ui as shiny_ui
from shiny_kit.ui import Icon


@dataclass(frozen=True, slots=True)
class PackageMetadata:
    """Resolved package metadata for `seroepi` extracted from `importlib.metadata`.

    Attributes:
        name (str): Distribution package name (`"seroepi"`).
        version (str): Full PEP 440 version string (including any `.dev` or `+g...` suffix).
        summary (str): One-line package summary description.
        author (str): Primary author name.
        author_email (str): Primary author contact email address.
        author_url (str): Author website URL.
        homepage (str): Canonical documentation / homepage URL.
        documentation (str): Documentation site URL.
        repository (str): Source code GitHub repository URL.
        bug_tracker (str): Issue tracker URL.
        changelog (str): Changelog URL.
        pypi_url (str): PyPI package project URL.
        license_spdx (str): Short license identifier (`"GPL-3.0"`).
    """

    name: str
    version: str
    summary: str
    author: str
    author_email: str
    author_url: str
    homepage: str
    documentation: str
    repository: str
    bug_tracker: str
    changelog: str
    pypi_url: str
    license_spdx: str = "GPL-3.0"

    @property
    def short_version(self) -> str:
        """Returns the version string with any local `+g...` git build suffix removed for compact display."""
        return self.version.split("+", 1)[0]

    @classmethod
    @lru_cache(maxsize=4)
    def load(cls, package_name: str = "seroepi") -> "PackageMetadata":
        """Loads distribution metadata via `importlib.metadata` with a source-tree fallback.

        Args:
            package_name: Installed Python distribution name to inspect. Defaults to `"seroepi"`.

        Returns:
            Populated `PackageMetadata` instance.
        """
        name = package_name
        version = "0.1.0"
        summary = "A Python library for bacterial sero-epidemiology"
        author = "Tom Stanton"
        author_email = "tomdstanton@gmail.com"
        author_url = "https://tomdstanton.com"
        urls: dict[str, str] = {
            "homepage": "https://seroverse.github.io/seroepi/",
            "documentation": "https://seroverse.github.io/seroepi/",
            "repository": "https://github.com/seroverse/seroepi",
            "bug tracker": "https://github.com/seroverse/seroepi/issues",
            "changelog": "https://github.com/seroverse/seroepi/blob/master/CHANGELOG.md",
        }

        try:
            from seroepi._version import __version__ as vcs_version

            version = vcs_version
        except Exception:
            pass

        try:
            meta = importlib.metadata.metadata(package_name)
            name = meta.get("Name") or name
            version = meta.get("Version") or version
            summary = meta.get("Summary") or summary

            raw_author = meta.get("Author")
            raw_author_email = meta.get("Author-email")
            if raw_author_email:
                parsed_name, parsed_email = parseaddr(raw_author_email)
                if parsed_name:
                    author = parsed_name
                if parsed_email:
                    author_email = parsed_email
            if raw_author:
                author = raw_author

            project_urls = meta.get_all("Project-URL") or []
            for entry in project_urls:
                if "," in entry:
                    label, url_val = entry.split(",", 1)
                    urls[label.strip().lower()] = url_val.strip()
        except importlib.metadata.PackageNotFoundError:
            pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
            if pyproject_path.is_file():
                try:
                    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
                    proj = data.get("project", {})
                    name = str(proj.get("name", name))
                    summary = str(proj.get("description", summary))
                    authors = proj.get("authors", [])
                    if authors and isinstance(authors[0], dict):
                        author = str(authors[0].get("name", author))
                        author_email = str(authors[0].get("email", author_email))
                    for label, url_val in proj.get("urls", {}).items():
                        urls[str(label).strip().lower()] = str(url_val).strip()
                except Exception:
                    pass

        return cls(
            name=name,
            version=version,
            summary=summary,
            author=author,
            author_email=author_email,
            author_url=author_url,
            homepage=urls["homepage"],
            documentation=urls["documentation"],
            repository=urls["repository"],
            bug_tracker=urls["bug tracker"],
            changelog=urls["changelog"],
            pypi_url=f"https://pypi.org/project/{name}/",
        )


@dataclass(frozen=True, slots=True)
class EcosystemPillarSpec:
    """Specification for a SeroVerse ecosystem pillar pill in the application footer.

    Attributes:
        name (str): Display name of the SeroVerse pillar.
        slug (str): CSS modifier slug (`"seroverse"`, `"serogen"`, `"kbank"`, or `"seroepi"`).
        icon (str): Lucide icon identifier (`"orbit"`, `"dna"`, `"hexagon"`, or `"chart-network"`).
        url (str): Documentation or landing page URL.
        description (str): Tooltip summary of the pillar's role in the SeroVerse.
        is_active (bool): Whether this pillar corresponds to the running application.
    """

    name: str
    slug: str
    icon: str
    url: str
    description: str
    is_active: bool = False


SEROVERSE_PILLARS: tuple[EcosystemPillarSpec, ...] = (
    EcosystemPillarSpec(
        name="SeroVerse",
        slug="seroverse",
        icon="orbit",
        url="https://seroverse.github.io/",
        description="SeroVerse Hub — Unified bacterial surface polysaccharide genomics, glycomics & sero-epidemiology",
    ),
    EcosystemPillarSpec(
        name="SeroGen",
        slug="serogen",
        icon="dna",
        url="https://seroverse.github.io/SeroGen/",
        description="SeroGen — Genomic locus typing & biosynthetic gene cluster engine",
    ),
    EcosystemPillarSpec(
        name="KBank",
        slug="kbank",
        icon="hexagon",
        url="https://seroverse.github.io/KBank/",
        description="KBank — Curated surface polysaccharide structures & glycomic database",
    ),
    EcosystemPillarSpec(
        name="SeroEpi",
        slug="seroepi",
        icon="chart-network",
        url="https://seroverse.github.io/SeroEpi/",
        description="SeroEpi — Hierarchical Bayesian sero-epidemiology & multi-valent formulation design",
        is_active=True,
    ),
)


FOOTER_CSS = """
.sv-app-footer {
    margin-top: 2.25rem;
    padding: 0.9rem 1.25rem;
    border-top: 1px solid var(--sv-border, rgba(139, 92, 246, 0.28));
    background:
        radial-gradient(circle at 12% 50%, var(--sv-glow, rgba(139, 92, 246, 0.12)), transparent 50%),
        var(--sv-surface-tint, linear-gradient(145deg, rgba(124, 58, 237, 0.06), rgba(219, 39, 119, 0.03)));
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    font-size: 0.8rem;
}

.sv-app-footer-inner {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.85rem 1.25rem;
    max-width: 1600px;
    margin: 0 auto;
}

.sv-footer-left,
.sv-footer-right {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.55rem 0.75rem;
}

.sv-footer-pkg-brand {
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
    font-weight: 800;
    font-size: 0.88rem;
    color: var(--bs-body-color) !important;
    text-decoration: none !important;
}

.sv-footer-pkg-brand svg {
    color: var(--sv-accent, #7c3aed);
}

.sv-footer-pkg-brand:hover {
    color: var(--sv-accent, #7c3aed) !important;
}

.sv-footer-version-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    font-family: var(--bs-font-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.16rem 0.52rem;
    border-radius: 999px;
    background: var(--sv-accent-soft, rgba(139, 92, 246, 0.14));
    color: var(--sv-accent, #7c3aed);
    border: 1px solid var(--sv-border, rgba(139, 92, 246, 0.3));
    text-decoration: none !important;
}

.sv-footer-meta-sep {
    opacity: 0.35;
    user-select: none;
}

.sv-footer-author {
    color: var(--bs-secondary-color);
    font-size: 0.78rem;
}

.sv-footer-author a {
    color: var(--bs-body-color);
    font-weight: 600;
    text-decoration: none;
}

.sv-footer-author a:hover {
    color: var(--sv-accent, #7c3aed);
    text-decoration: underline;
}

.sv-footer-icon-links {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

.sv-footer-icon-link {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.22rem 0.55rem;
    border-radius: 0.45rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--bs-secondary-color) !important;
    background: rgba(148, 163, 184, 0.09);
    border: 1px solid rgba(148, 163, 184, 0.2);
    text-decoration: none !important;
    transition: color 0.15s ease, border-color 0.15s ease, background-color 0.15s ease, transform 0.15s ease;
}

.sv-footer-icon-link:hover {
    color: var(--sv-accent, #7c3aed) !important;
    border-color: var(--sv-border, rgba(139, 92, 246, 0.35));
    background: var(--sv-accent-soft, rgba(139, 92, 246, 0.12));
    transform: translateY(-1px);
}

.sv-footer-eco-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--bs-secondary-color);
    margin-right: 0.15rem;
}

.sv-footer-eco-pills {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem;
}

.sv-footer-eco-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.32rem;
    padding: 0.22rem 0.62rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-decoration: none !important;
    border: 1px solid rgba(148, 163, 184, 0.22);
    background: rgba(148, 163, 184, 0.08);
    color: var(--bs-body-color) !important;
    transition: transform 0.15s ease, border-color 0.15s ease, background-color 0.15s ease;
}

.sv-footer-eco-pill:hover {
    transform: translateY(-1px);
}

.sv-footer-eco-pill--seroverse svg {
    color: #0284c7;
}
[data-bs-theme="dark"] .sv-footer-eco-pill--seroverse svg {
    color: #38bdf8;
}
.sv-footer-eco-pill--seroverse:hover {
    border-color: rgba(56, 189, 248, 0.45);
    background: rgba(56, 189, 248, 0.12);
}

.sv-footer-eco-pill--serogen svg {
    color: #0d9488;
}
[data-bs-theme="dark"] .sv-footer-eco-pill--serogen svg {
    color: #2dd4bf;
}
.sv-footer-eco-pill--serogen:hover {
    border-color: rgba(20, 184, 166, 0.45);
    background: rgba(20, 184, 166, 0.12);
}

.sv-footer-eco-pill--kbank svg {
    color: #059669;
}
[data-bs-theme="dark"] .sv-footer-eco-pill--kbank svg {
    color: #34d399;
}
.sv-footer-eco-pill--kbank:hover {
    border-color: rgba(16, 185, 129, 0.45);
    background: rgba(16, 185, 129, 0.12);
}

.sv-footer-eco-pill--seroepi.is-active {
    color: #7c3aed !important;
    background: rgba(124, 58, 237, 0.12);
    border-color: rgba(124, 58, 237, 0.38);
}
[data-bs-theme="dark"] .sv-footer-eco-pill--seroepi.is-active {
    color: #c084fc !important;
    background: rgba(168, 85, 247, 0.16);
    border-color: rgba(168, 85, 247, 0.42);
}
.sv-footer-eco-pill--seroepi svg {
    color: currentColor;
}
"""


def github_brand_svg(size: int = 14) -> HTML:
    """Returns the official GitHub Octocat mark as an inline SVG inheriting `currentColor`."""
    return HTML(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        'viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
        '<path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 '
        "0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 "
        "1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 "
        "0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 "
        "1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 "
        "0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57 "
        'C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>'
    )


def pypi_brand_svg(size: int = 14) -> HTML:
    """Returns the official PyPI isometric cube mark as an inline SVG inheriting `currentColor`."""
    return HTML(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M12 2L3 7l9 5 9-5-9-5z"/>'
        '<path d="M3 17l9 5 9-5"/>'
        '<path d="M3 12l9 5 9-5"/>'
        '<path d="M12 12v10"/>'
        '<path d="M3 7v10"/>'
        '<path d="M21 7v10"/>'
        "</svg>"
    )


def app_footer(meta: PackageMetadata | None = None) -> Tag:
    """Renders the SeroVerse-themed 2-zone application footer from `PackageMetadata`.

    Left zone displays the `SeroEpi` brand mark (`chart-network`), version badge (`v{version}`),
    author attribution (`Tom Stanton`), and icon links to Documentation, GitHub, PyPI, and
    the Issue Tracker resolved via `importlib.metadata`. Right zone renders the colored
    SeroVerse ecosystem pills (`SeroVerse`, `SeroGen`, `KBank`, `SeroEpi`).

    Args:
        meta: Optional `PackageMetadata` override. Defaults to `PackageMetadata.load("seroepi")`.

    Returns:
        `shiny.ui.Tag` `<footer>` element suitable for `shiny.ui.page_navbar(footer=...)`.
    """
    pkg = meta or PackageMetadata.load("seroepi")

    left_zone = shiny_ui.div(
        shiny_ui.a(
            Icon.lucide("chart-network", size=16),
            shiny_ui.span("SeroEpi"),
            href=pkg.homepage,
            target="_blank",
            rel="noopener",
            title=pkg.summary,
            class_="sv-footer-pkg-brand",
        ),
        shiny_ui.a(
            f"v{pkg.short_version}",
            href=pkg.changelog,
            target="_blank",
            rel="noopener",
            title=f"{pkg.name} v{pkg.version} ({pkg.license_spdx} License) — View Changelog",
            class_="sv-footer-version-badge",
        ),
        shiny_ui.span("·", class_="sv-footer-meta-sep"),
        shiny_ui.span(
            "by ",
            shiny_ui.a(
                pkg.author,
                href=pkg.author_url,
                target="_blank",
                rel="noopener",
                title=f"{pkg.author} ({pkg.author_email})",
            ),
            class_="sv-footer-author",
        ),
        shiny_ui.span("·", class_="sv-footer-meta-sep"),
        shiny_ui.div(
            shiny_ui.a(
                Icon.lucide("book-open", size=13),
                shiny_ui.span("Docs"),
                href=pkg.documentation,
                target="_blank",
                rel="noopener",
                title=f"Documentation ({pkg.documentation})",
                class_="sv-footer-icon-link",
            ),
            shiny_ui.a(
                github_brand_svg(size=13),
                shiny_ui.span("GitHub"),
                href=pkg.repository,
                target="_blank",
                rel="noopener",
                title=f"GitHub Repository ({pkg.repository})",
                class_="sv-footer-icon-link",
            ),
            shiny_ui.a(
                pypi_brand_svg(size=13),
                shiny_ui.span("PyPI"),
                href=pkg.pypi_url,
                target="_blank",
                rel="noopener",
                title=f"PyPI Package ({pkg.pypi_url})",
                class_="sv-footer-icon-link",
            ),
            shiny_ui.a(
                Icon.lucide("bug", size=13),
                shiny_ui.span("Issues"),
                href=pkg.bug_tracker,
                target="_blank",
                rel="noopener",
                title=f"Bug Tracker ({pkg.bug_tracker})",
                class_="sv-footer-icon-link",
            ),
            class_="sv-footer-icon-links",
        ),
        class_="sv-footer-left",
    )

    pillar_nodes = [
        shiny_ui.a(
            Icon.lucide(pillar.icon, size=13),
            shiny_ui.span(pillar.name),
            href=pillar.url,
            target="_blank",
            rel="noopener",
            title=pillar.description,
            class_=(
                f"sv-footer-eco-pill sv-footer-eco-pill--{pillar.slug}" + (" is-active" if pillar.is_active else "")
            ),
        )
        for pillar in SEROVERSE_PILLARS
    ]

    right_zone = shiny_ui.div(
        shiny_ui.span("SeroVerse Ecosystem", class_="sv-footer-eco-label"),
        shiny_ui.div(*pillar_nodes, class_="sv-footer-eco-pills"),
        class_="sv-footer-right",
    )

    return shiny_ui.tags.footer(
        shiny_ui.tags.style(FOOTER_CSS),
        shiny_ui.div(
            left_zone,
            right_zone,
            class_="sv-app-footer-inner",
        ),
        class_="sv-app-footer",
    )


__all__ = [
    "EcosystemPillarSpec",
    "FOOTER_CSS",
    "PackageMetadata",
    "SEROVERSE_PILLARS",
    "app_footer",
    "github_brand_svg",
    "pypi_brand_svg",
]
