"""Full-bleed, theme-aware Hero & Home Tab module integrated with `BasePlotter`."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

import altair as alt
from htmltools import TagChild
from shiny import req
from shiny import ui as shiny_ui
from shiny_kit.component import Tab
from shiny_kit.decorators import reactive_calc
from shiny_kit.ui import Icon
from shinywidgets import output_widget

from seroepi.aggregate import Aggregator
from seroepi.app.caching import default_cache
from seroepi.app.decorators import bind_altair
from seroepi.app.models import SessionState
from seroepi.estimators.base import BaseEstimator
from seroepi.estimators.unpooled import UnpooledEstimator
from seroepi.examples import ExampleDataset, KlebnetNeonatalSepsis
from seroepi.plotting.bar import BarPlotter
from seroepi.plotting.base import BasePlotter

_ = KlebnetNeonatalSepsis  # TODO: What does ths do?

# ---------------------------------------------------------------------------
# 1. Data-Oriented Configuration & Reactive Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkflowStepSpec:
    """Static specification for a workflow card below the Hero section.

    Attributes:
        step_label (str): Card heading for the pipeline step.
        description (str): Body copy describing the step's capabilities.
        icon (str): Lucide icon name rendered in the card header. Defaults to `"sparkles"`.
        badge (str): Optional pill badge text rendered at the top-right of the card.
        target_tab (str | None): Optional target navbar tab ID associated with the step.
    """

    step_label: str
    description: str
    icon: str = "sparkles"
    badge: str = ""
    target_tab: str | None = None


@dataclass(frozen=True, slots=True)
class HeroConfig:
    """Configuration contract for the Hero & Home landing view.

    Attributes:
        badge_text (str): Pill badge text displayed above the headline.
        headline (str): Main gradient headline text.
        subtitle (str): Lead paragraph text beneath the headline.
        primary_cta_label (str): Label for the primary action button.
        secondary_cta_label (str): Label for the secondary demo-loading button.
        preview_title (str): Header title of the glassmorphic live chart card.
        workflow_steps (tuple[WorkflowStepSpec, ...]): Sequence of workflow cards rendered below the fold.
    """

    badge_text: str
    headline: str
    subtitle: str
    primary_cta_label: str
    secondary_cta_label: str
    preview_title: str
    workflow_steps: tuple[WorkflowStepSpec, ...]


@dataclass(frozen=True, slots=True)
class HomeTabEvents:
    """Read-only reactive event streams emitted by `HomeTab.server()`.

    Attributes:
        launch_clicked (Callable[[], int]): Reactive getter returning the click count of the primary launch button.
        demo_clicked (Callable[[], int]): Reactive getter returning the click count of the demo cohort button.
    """

    launch_clicked: Callable[[], int]
    demo_clicked: Callable[[], int]


DEFAULT_HERO_CONFIG = HeroConfig(
    badge_text="Genomic Sero-Epidemiology & Vaccine Design",
    headline="Precision Antigen Selection from Genomic Surveillance",
    subtitle=(
        "Explore pathogen isolate collections, engineer compositional antigenic targets, "
        "fit hierarchical Bayesian and adjusted prevalence models, and evaluate "
        "multi-valent vaccine and phage-cocktail formulations."
    ),
    primary_cta_label="Explore Dataset & Targets",
    secondary_cta_label="Load Benchmark Cohort",
    preview_title="Interactive Antigen Prevalence Preview",
    workflow_steps=(
        WorkflowStepSpec(
            step_label="1. Dataset & Target Composer",
            description=(
                "Load genomic surveillance cohorts, apply semantic QC & variable filters, "
                "cluster pairwise distance matrices, enrich with UN/World Bank CountryData, "
                "and compose a cached repertoire of antigenic Targets."
            ),
            icon="database",
            badge="Ingest & Curate",
            target_tab="dataset_tab",
        ),
        WorkflowStepSpec(
            step_label="2. Stratified Prevalence Estimation",
            description=(
                "Aggregate isolate counts across geographic or temporal strata with optional "
                "adjustment variables, then fit Unpooled, Bootstrap, or Hierarchical "
                "Bayesian (MCMC / SVI) prevalence models."
            ),
            icon="trending-up",
            badge="Model & Visualize",
            target_tab="estimation_tab",
        ),
        WorkflowStepSpec(
            step_label="3. Interactive Epidemiological Visuals",
            description=(
                "Inspect prevalence estimates across Mapbox-style spatial choropleths, "
                "cumulative coverage curves, UpSet co-occurrence matrices, and back-to-back "
                "pyramid comparisons with vector SVG export."
            ),
            icon="shield-check",
            badge="Compare & Export",
            target_tab="estimation_tab",
        ),
    ),
)


# ---------------------------------------------------------------------------
# 2. Scoped Full-Bleed & Dark-Mode CSS ([data-bs-theme])
# ---------------------------------------------------------------------------

HERO_CSS = """
body {
    overflow-x: clip;
}

:root,
[data-bs-theme="light"] {
    --hero-bg-base: #f8fafc;
    --hero-radial-1: rgba(37, 99, 235, 0.12);
    --hero-radial-2: rgba(14, 165, 233, 0.10);
    --hero-grid-line: rgba(15, 23, 42, 0.045);
    --hero-border: rgba(15, 23, 42, 0.10);
    --hero-card-bg: rgba(255, 255, 255, 0.82);
    --hero-title-grad: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #0284c7 100%);
    --hero-badge-bg: rgba(37, 99, 235, 0.10);
    --hero-badge-text: #1d4ed8;
}

[data-bs-theme="dark"] {
    --hero-bg-base: #0b1120;
    --hero-radial-1: rgba(59, 130, 246, 0.20);
    --hero-radial-2: rgba(6, 182, 212, 0.14);
    --hero-grid-line: rgba(255, 255, 255, 0.045);
    --hero-border: rgba(255, 255, 255, 0.12);
    --hero-card-bg: rgba(15, 23, 42, 0.76);
    --hero-title-grad: linear-gradient(135deg, #f8fafc 0%, #60a5fa 55%, #38bdf8 100%);
    --hero-badge-bg: rgba(59, 130, 246, 0.18);
    --hero-badge-text: #93c5fd;
}

.hero-full-bleed {
    width: 100vw;
    margin-inline: calc(50% - 50vw);
    margin-top: -1rem;
    position: relative;
    background-color: var(--hero-bg-base);
    background-image:
        radial-gradient(circle at 14% 22%, var(--hero-radial-1) 0%, transparent 48%),
        radial-gradient(circle at 86% 78%, var(--hero-radial-2) 0%, transparent 52%),
        linear-gradient(to right, var(--hero-grid-line) 1px, transparent 1px),
        linear-gradient(to bottom, var(--hero-grid-line) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px;
    border-bottom: 1px solid var(--hero-border);
}

.hero-cta-group {
    display: inline-grid;
    grid-auto-flow: column;
    grid-auto-columns: 1fr;
    gap: 1rem;
}

@media (max-width: 576px) {
    .hero-cta-group {
        display: grid;
        grid-auto-flow: row;
        width: 100%;
    }
}

.hero-headline {
    background: var(--hero-title-grad);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    line-height: 1.15;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    background: var(--hero-badge-bg);
    color: var(--hero-badge-text);
    border: 1px solid var(--hero-border);
}

.hero-glass-card {
    background: var(--hero-card-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--hero-border);
    border-radius: 1rem;
}

.hero-workflow-card {
    background: var(--hero-card-bg);
    border: 1px solid var(--hero-border);
    border-radius: 0.875rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.hero-workflow-card:hover {
    transform: translateY(-2px);
}
"""


# ---------------------------------------------------------------------------
# 3. OOP HomeTab Module
# ---------------------------------------------------------------------------


class HomeTab(Tab[SessionState]):
    """Home.

    Full-bleed, theme-aware landing tab featuring a live interactive `BarPlotter`
    specimen, vector SVG download, and end-to-end epidemiological workflow cards.

    Attributes:
        icon (Tag): Lucide `"house"` icon displayed in the top navbar.
        config (ClassVar[HeroConfig]): Default class-level `HeroConfig` specification.
        instance_config (HeroConfig): Instance-level `HeroConfig` specification.
        hero_plotter (BasePlotter): `BasePlotter` instance used to render the live specimen chart.
    """

    icon = Icon.lucide("house")
    config: ClassVar[HeroConfig] = DEFAULT_HERO_CONFIG
    _DEFAULT_ESTIMATOR: ClassVar[BaseEstimator | None] = None
    _DEFAULT_CHART_CACHE: ClassVar[dict[tuple[Literal["light", "dark"], bool], alt.TopLevelMixin]] = {}

    def __init__(
        self,
        *,
        config: HeroConfig | None = None,
        hero_plotter: BasePlotter | None = None,
        color_mode: Callable[[], str] | None = None,
        estimator_provider: Callable[[], BaseEstimator | None] | BaseEstimator | None = None,
    ) -> None:
        """Initializes the `HomeTab` instance.

        Args:
            config: Optional custom `HeroConfig` overriding `cls.config`.
            hero_plotter: Optional custom `BasePlotter` used for the specimen preview card.
            color_mode: Optional callable returning `"light"` or `"dark"`.
            estimator_provider: Optional `BaseEstimator` or callable returning the estimator to preview.
        """
        super().__init__()
        if config is not None:
            self.instance_config = config
        else:
            self.instance_config = self.config
        self._has_custom_plotter = hero_plotter is not None
        self.hero_plotter: BasePlotter = hero_plotter or self._build_default_plotter()
        self._color_mode_fn = color_mode
        self._estimator_provider = estimator_provider

    @staticmethod
    def _build_default_plotter() -> BarPlotter:
        return BarPlotter(
            top_n=10,
            bar_color="#3b82f6",
            width=460,
            height=300,
        )

    @classmethod
    def default_hero_estimator(cls) -> BaseEstimator:
        """Returns a process-cached benchmark `UnpooledEstimator` fitted on the default `ExampleDataset`'s primary `Trait`.

        Returns:
            Fitted `BaseEstimator` instance for the default benchmark cohort.
        """
        if cls._DEFAULT_ESTIMATOR is None:
            example_cls = ExampleDataset.__subclasses__()[0]
            ds = example_cls.load()
            primary_trait = ds.pure_traits[0] if ds.pure_traits else next(iter(ds.variables.values()))
            agg = default_cache.fit_or_load(Aggregator(target=primary_trait), ds)
            est = default_cache.fit_or_load(UnpooledEstimator(name=f"Wilson ({ds.name})"), agg)
            cls._DEFAULT_ESTIMATOR = est
        return cls._DEFAULT_ESTIMATOR

    @classmethod
    def default_hero_chart(
        cls,
        theme_mode: Literal["light", "dark"] = "light",
        show_ci: bool = True,
    ) -> alt.TopLevelMixin:
        """Returns the process-cached default Hero Altair chart specimen for `(theme_mode, show_ci)`.

        Args:
            theme_mode: Visual theme palette (`"light"` or `"dark"`).
            show_ci: Whether to display 95% confidence/credible interval error bars.

        Returns:
            Configured Altair chart object.
        """
        key = (theme_mode, bool(show_ci))
        cached = cls._DEFAULT_CHART_CACHE.get(key)
        if cached is None:
            plotter = cls._build_default_plotter()
            plotter.show_error_bars = show_ci
            plotter.with_theme(theme_mode)
            cached = plotter.plot(cls.default_hero_estimator(), show_error_bars=show_ci)
            cls._DEFAULT_CHART_CACHE[key] = cached
        return cached

    @classmethod
    def _render_ui(cls) -> TagChild:
        cfg = cls.config
        return shiny_ui.div(
            shiny_ui.head_content(shiny_ui.tags.style(HERO_CSS)),
            # --- Full-Bleed Hero Section ---
            shiny_ui.div(
                shiny_ui.div(
                    shiny_ui.layout_columns(
                        # Left Column: Badge, Headline, Subtitle, CTAs
                        shiny_ui.div(
                            shiny_ui.span(
                                Icon.lucide("sparkles", size=14),
                                cfg.badge_text,
                                class_="hero-badge rounded-pill mb-3",
                            ),
                            shiny_ui.h1(
                                cfg.headline,
                                class_="display-5 fw-bold hero-headline mb-3",
                            ),
                            shiny_ui.p(
                                cfg.subtitle,
                                class_="lead text-body-secondary mb-4",
                            ),
                            shiny_ui.div(
                                shiny_ui.input_action_button(
                                    "btn_launch",
                                    cfg.primary_cta_label,
                                    icon=Icon.lucide("rocket", size=16),
                                    class_="btn-primary btn-lg px-4 shadow-sm",
                                ),
                                shiny_ui.input_action_button(
                                    "btn_load_demo",
                                    cfg.secondary_cta_label,
                                    icon=Icon.lucide("play", size=16),
                                    class_="btn-outline-primary btn-lg px-4",
                                ),
                                class_="hero-cta-group align-items-center",
                                # class_="d-flex flex-wrap gap-3 align-items-center",
                            ),
                        ),
                        # Right Column: Glassmorphic Live BasePlotter Specimen Card
                        shiny_ui.div(
                            shiny_ui.div(
                                shiny_ui.div(
                                    Icon.lucide("chart-bar", size=16, class_="text-primary me-1"),
                                    shiny_ui.span(cfg.preview_title, class_="fw-semibold small"),
                                    class_="d-flex align-items-center",
                                ),
                                shiny_ui.div(
                                    shiny_ui.div(
                                        shiny_ui.input_switch("toggle_ci", "95% CI", value=True, width="auto"),
                                        class_="mb-0 d-flex align-items-center",
                                    ),
                                    class_="d-flex align-items-center gap-3",
                                ),
                                class_="d-flex flex-wrap justify-content-between align-items-center pb-2 mb-2 border-bottom",
                            ),
                            output_widget(
                                "hero_altair_chart",
                                width="100%",
                                height="340px",
                                fill=True,
                                fillable=True,
                            ),
                            class_="hero-glass-card p-3 p-md-4 shadow-lg",
                        ),
                        col_widths={"sm": (12, 12), "lg": (6, 6)},
                        class_="align-items-center g-5",
                        fill=False,
                    ),
                    class_="container py-5",
                ),
                class_="hero-full-bleed",
            ),
            # --- Below the Fold: End-to-End Workflow Architecture ---
            shiny_ui.div(
                shiny_ui.div(
                    shiny_ui.h4("End-to-End Epidemiological Pipeline", class_="fw-bold mb-1"),
                    shiny_ui.p(
                        "Each workspace unlocks progressively as you curate datasets, engineer targets, and train estimators.",
                        class_="text-body-secondary mb-4",
                    ),
                ),
                shiny_ui.layout_column_wrap(
                    *[
                        shiny_ui.div(
                            shiny_ui.div(
                                shiny_ui.span(
                                    Icon.lucide(step.icon, size=18, class_="text-primary"),
                                    class_="p-2 rounded-3 bg-primary-subtle d-inline-flex align-items-center justify-content-center",
                                ),
                                shiny_ui.span(
                                    step.badge,
                                    class_="badge bg-secondary-subtle text-secondary-emphasis rounded-pill px-2 py-1",
                                )
                                if step.badge
                                else None,
                                class_="d-flex justify-content-between align-items-center mb-3",
                            ),
                            shiny_ui.h6(step.step_label, class_="fw-bold mb-2"),
                            shiny_ui.p(step.description, class_="text-body-secondary small mb-0"),
                            class_="hero-workflow-card p-4 h-100 shadow-sm",
                        )
                        for step in cfg.workflow_steps
                    ],
                    width=1 / 3,
                    class_="g-4",
                    fill=False,
                ),
                class_="container py-5",
            ),
        )

    def _resolve_active_theme(self) -> Literal["light", "dark"]:
        if self._color_mode_fn is not None:
            try:
                mode_val = self._color_mode_fn()
                return "dark" if mode_val == "dark" else "light"
            except Exception:
                pass
        return "dark" if self.state.color_mode == "dark" else "light"

    def _resolve_hero_estimator(self) -> BaseEstimator:
        est: BaseEstimator | None = None
        if isinstance(self._estimator_provider, BaseEstimator):
            est = self._estimator_provider
        elif callable(self._estimator_provider):
            est = self._estimator_provider()
        elif self.state.active_estimator is not None:
            est = self.state.active_estimator

        if est is None:
            est = self.default_hero_estimator()

        if est.n_strata > 0 and not self.hero_plotter.can_plot(est):
            est = est.pool_strata()
        return est

    @reactive_calc
    def hero_chart(self) -> alt.TopLevelMixin:
        """Reactively builds or retrieves the theme-synced Hero Altair chart specimen.

        Returns:
            Configured Altair chart for the active or default estimator.
        """
        estimator = self._resolve_hero_estimator()
        req(self.hero_plotter.can_plot(estimator))
        show_ci = bool(self.val("toggle_ci", default=True))
        theme_mode = self._resolve_active_theme()
        if not self._has_custom_plotter and estimator is self.default_hero_estimator():
            return self.default_hero_chart(theme_mode=theme_mode, show_ci=show_ci)
        self.hero_plotter.show_error_bars = show_ci
        self.hero_plotter.with_theme(theme_mode)
        return self.hero_plotter.plot(estimator, show_error_bars=show_ci)

    @bind_altair("hero_altair_chart")
    def render_hero_altair_chart(self) -> alt.TopLevelMixin:
        """Renders the reactive `hero_chart()` into the `#hero_altair_chart` widget output.

        Returns:
            Configured Altair chart.
        """
        return self.hero_chart()

    def server(
        self,
        id_: str,
        *,
        color_mode: Callable[[], str] | None = None,
        estimator_provider: Callable[[], BaseEstimator | None] | BaseEstimator | None = None,
    ) -> HomeTabEvents:
        """Binds server logic for the Home tab and returns `HomeTabEvents` streams.

        Args:
            id_: Shiny module namespace ID.
            color_mode: Optional reactive getter returning `"light"` or `"dark"`.
            estimator_provider: Optional estimator or reactive getter overriding the active estimator.

        Returns:
            `HomeTabEvents` containing `launch_clicked` and `demo_clicked` reactive getters.
        """
        if color_mode is not None:
            self._color_mode_fn = color_mode
        if estimator_provider is not None:
            self._estimator_provider = estimator_provider
        super().server(id_)

        def _launch_clicked() -> int:
            return int(self.val("btn_launch", default=0) or 0)

        def _demo_clicked() -> int:
            return int(self.val("btn_load_demo", default=0) or 0)

        return HomeTabEvents(
            launch_clicked=_launch_clicked,
            demo_clicked=_demo_clicked,
        )
