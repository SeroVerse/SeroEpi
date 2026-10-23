"""Full-bleed, SeroVerse-themed Hero & Home Tab module integrated with `BasePlotter`."""

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
from seroepi.dataset import Dataset
from seroepi.estimators.base import BaseEstimator
from seroepi.estimators.unpooled import UnpooledEstimator
from seroepi.plotting.bar import BarPlotter
from seroepi.plotting.base import BasePlotter

# ---------------------------------------------------------------------------
# 1. Data-Oriented Configuration & Reactive Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeroStatSpec:
    """Static specification for a metric tile in the `.sv-stat-strip` hero strip.

    Attributes:
        value (str): Highlighted headline metric value (rendered with gradient text).
        label (str): Descriptive caption displayed beneath `value`.
    """

    value: str
    label: str


@dataclass(frozen=True, slots=True)
class WorkflowStepSpec:
    """Static specification for a SeroVerse pillar/workflow card below the Hero section.

    Attributes:
        step_label (str): Card heading for the pipeline step.
        description (str): Body copy describing the step's capabilities.
        icon (str): Lucide icon name rendered in the card header. Defaults to `"sparkles"`.
        badge (str): Optional uppercase pill badge text rendered at the top-right of the card.
        metrics (tuple[str, ...]): Optional metric pills displayed inside the card footer.
        target_tab (str | None): Optional target navbar tab ID associated with the step.
    """

    step_label: str
    description: str
    icon: str = "sparkles"
    badge: str = ""
    metrics: tuple[str, ...] = ()
    target_tab: str | None = None


@dataclass(frozen=True, slots=True)
class HeroConfig:
    """Configuration contract for the SeroVerse Hero & Home landing view.

    Attributes:
        badge_text (str): Eyebrow pill text displayed above the headline.
        headline (str): Main gradient headline text.
        subtitle (str): Lead paragraph text beneath the headline.
        primary_cta_label (str): Label for the primary action button.
        secondary_cta_label (str): Label for the secondary demo-loading button.
        preview_title (str): Header title of the glassmorphic live chart card.
        stats (tuple[HeroStatSpec, ...]): Sequence of metrics rendered in the `.sv-stat-strip` bar.
        workflow_steps (tuple[WorkflowStepSpec, ...]): Sequence of workflow cards rendered below the fold.
    """

    badge_text: str
    headline: str
    subtitle: str
    primary_cta_label: str
    secondary_cta_label: str
    preview_title: str
    workflow_steps: tuple[WorkflowStepSpec, ...]
    stats: tuple[HeroStatSpec, ...] = ()


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
    badge_text="SeroEpi",
    headline="Precision Target Selection from Genomic Surveillance",
    subtitle=(
        "Explore pathogen isolate collections, engineer compositional or binary targets, "
        "fit JAX/NumPyro hierarchical Bayesian prevalence models, and optimize "
        "cross-validated multi-valent formulations."
    ),
    primary_cta_label="Explore Dataset & Targets",
    secondary_cta_label="Load Benchmark Cohort",
    preview_title="Interactive Target Prevalence Preview",
    stats=(
        HeroStatSpec(
            value="981 Isolates",
            label="Curated KLEBNET neonatal sepsis benchmark cohort with pairwise SNP distances",
        ),
        HeroStatSpec(
            value="MCMC & SVI",
            label="JAX/NumPyro hierarchical partial-pooling & analytical/bootstrap estimators",
        ),
        HeroStatSpec(
            value="LOO-CV & FDR",
            label="Leave-One-Stratum-Out formulation design & permutation differential testing",
        ),
        HeroStatSpec(
            value="9 Plotters",
            label="Publication-grade Altair choropleths, Pareto curves, UpSet & bump charts",
        ),
    ),
    workflow_steps=(
        WorkflowStepSpec(
            step_label="1. Dataset & Target Composer",
            description=(
                "Ingest Kleborate tables or live Pathogenwatch cohorts, apply semantic QC filters, "
                "de-duplicate clonal outbreaks via SNP/spatiotemporal graph clustering, enrich with "
                "UN/World Bank CountryData, and compose multi-variable Targets."
            ),
            icon="database",
            badge="1 · Ingest & Curate",
            metrics=("Pathogenwatch API", "Outbreak Clustering", "World Bank Enrichment"),
            target_tab="dataset_tab",
        ),
        WorkflowStepSpec(
            step_label="2. Stratified Prevalence Estimation",
            description=(
                "Aggregate binary or compositional isolate counts across geographic or clinical "
                "strata with zero-padding, then fit Unpooled (Wilson, Jeffreys, Dirichlet) or "
                "Hierarchical Bayesian (NumPyro NUTS MCMC & SVI) partial-pooling models."
            ),
            icon="chart-network",
            badge="2 · Model & Visualize",
            metrics=("NumPyro NUTS & SVI", "Spatial Choropleths", "UpSet & Pyramid Plots"),
            target_tab="estimation_tab",
        ),
        WorkflowStepSpec(
            step_label="3. Multi-Valent Formulation & Coverage",
            description=(
                "Optimize K-valent target formulations using greedy coverage "
                "maximization, evaluate geographic generalizability via Leave-One-Out CV, and "
                "test differential stratum enrichment with BH/BY False Discovery Rate control."
            ),
            icon="shield-check",
            badge="3 · Optimize & Validate",
            metrics=("Pareto Coverage Curves", "LOO-CV Rank Stability", "Volcano FDR Testing"),
            target_tab="formulation_tab",
        ),
    ),
)


# ---------------------------------------------------------------------------
# 2. SeroVerse / SeroEpi Unified Theme & Hero CSS ([data-bs-theme])
# ---------------------------------------------------------------------------

HERO_CSS = """
body {
    overflow-x: clip;
}

:root,
[data-bs-theme="light"] {
    --sv-gradient: linear-gradient(135deg, #7c3aed 0%, #6366f1 52%, #db2777 100%);
    --sv-glow: rgba(139, 92, 246, 0.16);
    --sv-border: rgba(139, 92, 246, 0.30);
    --sv-surface-tint: linear-gradient(145deg, rgba(124, 58, 237, 0.09), rgba(219, 39, 119, 0.05));
    --sv-accent: #7c3aed;
    --sv-accent-soft: rgba(139, 92, 246, 0.12);
    --sv-card-bg: rgba(255, 255, 255, 0.86);
    --sv-stat-bg: rgba(15, 23, 42, 0.035);
    --bs-primary: #7c3aed;
    --bs-primary-rgb: 124, 58, 237;
    --bs-link-color: #7c3aed;
    --bs-link-hover-color: #6d28d9;
}

[data-bs-theme="dark"] {
    --sv-gradient: linear-gradient(135deg, #a78bfa 0%, #818cf8 52%, #f472b6 100%);
    --sv-glow: rgba(167, 139, 250, 0.20);
    --sv-border: rgba(167, 139, 250, 0.32);
    --sv-surface-tint: linear-gradient(145deg, rgba(124, 58, 237, 0.16), rgba(219, 39, 119, 0.10));
    --sv-accent: #c084fc;
    --sv-accent-soft: rgba(168, 85, 247, 0.16);
    --sv-card-bg: rgba(15, 23, 42, 0.78);
    --sv-stat-bg: rgba(15, 23, 42, 0.58);
    --bs-body-bg: #0b1120;
    --bs-tertiary-bg: #111827;
    --bs-primary: #8b5cf6;
    --bs-primary-rgb: 139, 92, 246;
}

[data-bs-theme="dark"] .bg-light {
    background-color: var(--bs-tertiary-bg) !important;
    color: var(--bs-body-color) !important;
}

[data-bs-theme="dark"] .text-dark {
    color: var(--bs-body-color) !important;
}

.navbar-brand .sv-brand-icon svg {
    color: var(--sv-accent);
}

.sv-hero,
.serogen-hero {
    position: relative;
    padding: 2.4rem 2.2rem;
    margin-top: 0.75rem;
    margin-bottom: 1.5rem;
    border-radius: 1rem;
    background:
        radial-gradient(circle at 85% 15%, var(--sv-glow), transparent 55%),
        var(--sv-surface-tint);
    border: 1px solid var(--sv-border);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    container-type: inline-size;
}

.sv-hero-eyebrow,
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.42rem;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 0.32rem 0.78rem;
    margin-bottom: 0.9rem;
    border-radius: 999px;
    background: var(--sv-accent-soft);
    color: var(--sv-accent);
    border: 1px solid var(--sv-border);
}

.hero-headline {
    background: var(--sv-gradient);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    font-weight: 800;
    letter-spacing: -0.025em;
    line-height: 1.15;
}

.hero-cta-group {
    display: inline-grid;
    grid-template-columns: 1fr;
    gap: 0.75rem;
}

.hero-cta-group > .btn {
    width: 100%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

@container (min-width: 1120px) {
    .hero-cta-group {
        grid-template-columns: repeat(2, 1fr);
    }
}

.sv-btn-primary.btn {
    color: #ffffff !important;
    background: linear-gradient(135deg, #7c3aed 0%, #6366f1 52%, #db2777 100%) !important;
    border: none !important;
    border-radius: 0.6rem;
    font-weight: 600;
    box-shadow: 0 6px 18px -5px var(--sv-glow);
    transition: transform 0.16s ease, box-shadow 0.16s ease, opacity 0.16s ease;
}

.sv-btn-primary.btn:hover {
    transform: translateY(-1px);
    opacity: 0.95;
}

.sv-btn-secondary.btn {
    color: var(--bs-body-color) !important;
    background: rgba(148, 163, 184, 0.10) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: 0.6rem;
    font-weight: 600;
    transition: transform 0.16s ease, border-color 0.16s ease, background-color 0.16s ease;
}

.sv-btn-secondary.btn:hover {
    transform: translateY(-1px);
    border-color: var(--sv-accent) !important;
    background: var(--sv-accent-soft) !important;
}

.hero-glass-card {
    background: var(--sv-card-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--sv-border);
    border-radius: 0.9rem;
    box-shadow: 0 14px 34px -12px var(--sv-glow);
}

.sv-stat-strip {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.75rem;
    margin-top: 1.75rem;
    padding-top: 1.35rem;
    border-top: 1px solid rgba(148, 163, 184, 0.20);
}

@media (min-width: 640px) {
    .sv-stat-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (min-width: 992px) {
    .sv-stat-strip {
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }
}

.sv-stat-item {
    padding: 0.7rem 0.9rem;
    border-radius: 0.65rem;
    background: var(--sv-stat-bg);
    border: 1px solid rgba(148, 163, 184, 0.18);
}

.sv-stat-value {
    display: block;
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: var(--sv-gradient);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}

.sv-stat-label {
    display: block;
    font-size: 0.76rem;
    opacity: 0.84;
    margin-top: 0.15rem;
    line-height: 1.4;
}

.sv-pillar-card,
.hero-workflow-card {
    position: relative;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 1.45rem 1.4rem;
    border-radius: 0.9rem;
    border: 1px solid rgba(148, 163, 184, 0.22);
    background: var(--sv-card-bg);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    overflow: hidden;
}

.sv-pillar-card::before,
.hero-workflow-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--sv-gradient);
}

.sv-pillar-card:hover,
.hero-workflow-card:hover {
    transform: translateY(-3px);
    border-color: var(--sv-border);
    box-shadow: 0 14px 32px -12px var(--sv-glow);
}

.sv-pillar-badge {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.22rem 0.58rem;
    border-radius: 999px;
    background: var(--sv-accent-soft);
    color: var(--sv-accent);
    border: 1px solid var(--sv-border);
}

.sv-pillar-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 1rem;
    padding-top: 0.85rem;
    border-top: 1px solid rgba(148, 163, 184, 0.16);
}

.sv-metric-pill {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.18rem 0.54rem;
    border-radius: 0.4rem;
    background: rgba(148, 163, 184, 0.10);
    border: 1px solid rgba(148, 163, 184, 0.18);
}
"""


# ---------------------------------------------------------------------------
# 3. OOP HomeTab Module
# ---------------------------------------------------------------------------


class HomeTab(Tab[SessionState]):
    """Home.

    SeroVerse-themed landing tab featuring a glassmorphic `.sv-hero` banner, live
    interactive `BarPlotter` specimen, `.sv-stat-strip` metrics, and `.sv-pillar-card`
    epidemiological workflow cards.

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
            bar_color="#8b5cf6",
            width=460,
            height=300,
        )

    @classmethod
    def default_hero_estimator(cls) -> BaseEstimator:
        """Returns a process-cached benchmark `UnpooledEstimator` fitted on `Dataset.example()`'s primary `Trait`.

        Returns:
            Fitted `BaseEstimator` instance for the default benchmark cohort.
        """
        if cls._DEFAULT_ESTIMATOR is None:
            ds = Dataset.example(with_distances=False)
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
            shiny_ui.div(
                # --- SeroVerse Glassmorphic Hero Card (.sv-hero) ---
                shiny_ui.div(
                    shiny_ui.layout_columns(
                        # Left Column: Gradient Headline, Subtitle, CTAs
                        shiny_ui.div(
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
                                    class_="sv-btn-primary btn-lg px-4",
                                ),
                                shiny_ui.input_action_button(
                                    "btn_load_demo",
                                    cfg.secondary_cta_label,
                                    icon=Icon.lucide("play", size=16),
                                    class_="sv-btn-secondary btn-lg px-4",
                                ),
                                class_="hero-cta-group",
                            ),
                        ),
                        # Right Column: Glassmorphic Live BasePlotter Specimen Card
                        shiny_ui.div(
                            shiny_ui.div(
                                shiny_ui.div(
                                    Icon.lucide("chart-network", size=16, class_="text-primary me-2"),
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
                                height="320px",
                                fill=True,
                                fillable=True,
                            ),
                            class_="hero-glass-card p-3 p-md-4",
                        ),
                        col_widths={"sm": (12, 12), "lg": (6, 6)},
                        class_="align-items-center g-4",
                        fill=False,
                    ),
                    # --- Hero Stat Strip (.sv-stat-strip) ---
                    shiny_ui.div(
                        *[
                            shiny_ui.div(
                                shiny_ui.span(stat.value, class_="sv-stat-value"),
                                shiny_ui.span(stat.label, class_="sv-stat-label"),
                                class_="sv-stat-item",
                            )
                            for stat in cfg.stats
                        ],
                        class_="sv-stat-strip",
                    )
                    if cfg.stats
                    else None,
                    class_="sv-hero serogen-hero",
                ),
                # --- Below the Fold: End-to-End Workflow Architecture (.sv-pillar-card) ---
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
                                    shiny_ui.div(
                                        shiny_ui.span(
                                            Icon.lucide(step.icon, size=18),
                                            style="color: var(--sv-accent); background: var(--sv-accent-soft);",
                                            class_="p-2 rounded-3 d-inline-flex align-items-center justify-content-center",
                                        ),
                                        shiny_ui.span(
                                            step.badge,
                                            class_="sv-pillar-badge",
                                        )
                                        if step.badge
                                        else None,
                                        class_="d-flex justify-content-between align-items-center mb-3",
                                    ),
                                    shiny_ui.h6(step.step_label, class_="fw-bold mb-2"),
                                    shiny_ui.p(step.description, class_="text-body-secondary small mb-0"),
                                ),
                                shiny_ui.div(
                                    *[shiny_ui.span(m, class_="sv-metric-pill") for m in step.metrics],
                                    class_="sv-pillar-metrics",
                                )
                                if step.metrics
                                else None,
                                class_="sv-pillar-card hero-workflow-card h-100",
                            )
                            for step in cfg.workflow_steps
                        ],
                        width=1 / 3,
                        class_="g-4",
                        fill=False,
                    ),
                    class_="py-3",
                ),
                class_="container py-3",
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
