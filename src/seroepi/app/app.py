"""Main PyShiny application entrypoint for SeroEpi."""

import os
import sys
import threading
from pathlib import Path

from htmltools import TagChild
from shiny import App, Inputs, Outputs, Session, reactive, render, req
from shiny import ui as shiny_ui
from shiny_kit.ui import Icon

_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from seroepi.aggregate import Aggregator  # noqa: E402
from seroepi.app.dataset.loader import DatasetLoader  # noqa: E402
from seroepi.app.dataset.tab import DatasetTab  # noqa: E402
from seroepi.app.estimation import EstimationTab  # noqa: E402
from seroepi.app.footer import app_footer  # noqa: E402
from seroepi.app.formulation import FormulationTab  # noqa: E402
from seroepi.app.home import HomeTab  # noqa: E402
from seroepi.app.models import SessionState  # noqa: E402
from seroepi.country import CountryData  # noqa: E402
from seroepi.estimators.formulation import Formulation  # noqa: E402
from seroepi.estimators.unpooled import UnpooledEstimator  # noqa: E402
from seroepi.variables import Spatial, Target  # noqa: E402


def app_ui() -> shiny_ui.Tag:
    """Constructs the top-level navbar UI layout for the SeroEpi application.

    Composes `HomeTab`, `DatasetTab`, `EstimationTab`, and `FormulationTab` inside
    a `shiny.ui.page_navbar` container alongside reactive artifact registry badges,
    a Bootstrap light/dark mode switch, and the `importlib.metadata`-backed
    SeroVerse footer.

    Returns:
        Top-level `shiny.ui.Tag` representing the full application UI.
    """
    return shiny_ui.page_navbar(
        HomeTab.ui("home_tab"),
        DatasetTab.ui("dataset_tab"),
        EstimationTab.ui("estimation_tab"),
        FormulationTab.ui("formulation_tab"),
        shiny_ui.nav_spacer(),
        shiny_ui.nav_control(
            shiny_ui.output_ui("navbar_registry_badges"),
        ),
        shiny_ui.nav_control(
            shiny_ui.div(
                shiny_ui.input_dark_mode(id="color_mode", mode="light"),
                class_="d-flex align-items-center ps-1 pe-2",
            ),
        ),
        id="main_nav",
        title=shiny_ui.span(
            shiny_ui.span(Icon.lucide("chart-network", size=20), class_="sv-brand-icon d-inline-flex"),
            shiny_ui.strong("SeroEpi"),
            class_="d-inline-flex align-items-center gap-1",
        ),
        footer=app_footer(),
        fillable=False,
    )


def app_server(input: Inputs, output: Outputs, session: Session) -> None:
    """Binds top-level session state and page modules.

    Initializes the session-scoped `SessionState`, synchronizes light/dark color
    mode, mounts the four top-level `Tab` modules, and wires Hero landing page CTA
    events (`launch_clicked` and `demo_clicked`).

    Args:
        input: Top-level Shiny `Inputs` object.
        output: Top-level Shiny `Outputs` object.
        session: Active Shiny `Session` instance.
    """
    state = SessionState.from_session(session)

    @reactive.effect
    def _sync_color_mode() -> None:
        try:
            mode_val = input.color_mode()
        except Exception:
            mode_val = "light"
        state.color_mode = "dark" if mode_val == "dark" else "light"

    home_tab = HomeTab()
    home_events = home_tab.server("home_tab", color_mode=input.color_mode)

    dataset_tab = DatasetTab()
    dataset_tab.server("dataset_tab")

    estimation_tab = EstimationTab()
    estimation_tab.server("estimation_tab")

    formulation_tab = FormulationTab()
    formulation_tab.server("formulation_tab")

    @reactive.effect
    @reactive.event(home_events.launch_clicked, ignore_init=True)
    def _on_hero_launch() -> None:
        req(home_events.launch_clicked() > 0)
        shiny_ui.update_navset("main_nav", selected="dataset_tab", session=session)

    @reactive.effect
    @reactive.event(home_events.demo_clicked, ignore_init=True)
    def _on_hero_load_demo() -> None:
        req(home_events.demo_clicked() > 0)
        example_cls = next(iter(DatasetLoader.example_registry().values()))
        ds = example_cls.load() if example_cls.is_built() else example_cls.build()
        state.register_dataset(ds, reset_filters=True)
        traits = ds.pure_traits
        for idx, trait in enumerate(traits):
            tgt = Target.coerce(trait)
            state.targets.register(tgt.name, tgt, set_active=(idx == 0))
        if traits:
            primary_trait = traits[0]
            primary_spatial = next((v for v in ds.variables.values() if isinstance(v, Spatial)), None)
            strata_arg = [primary_spatial] if primary_spatial is not None else None
            agg = state.cache.fit_or_load(Aggregator(target=primary_trait, strata=strata_arg), ds)
            state.register_aggregator(agg)
            est = state.cache.fit_or_load(UnpooledEstimator(), agg)
            state.register_estimator(est)
            form = state.cache.fit_or_load(Formulation(estimator=est, valency=6), agg)
            state.register_formulation(form)
        shiny_ui.notification_show(
            f"Loaded benchmark '{ds.name}' ({len(ds):,} isolates, {len(traits)} targets).",
            type="message",
            duration=4,
            session=session,
        )
        shiny_ui.update_navset("main_nav", selected="dataset_tab", session=session)

    @render.ui
    def navbar_registry_badges() -> TagChild:
        return state.render_navbar_badges()

    output.navbar_registry_badges = navbar_registry_badges  # ty: ignore[unresolved-attribute]


def _warm_reference_assets() -> None:
    """Warms `HomeTab.default_hero_estimator()` and `CountryData` in a background daemon thread on worker boot."""
    if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
        return

    def _worker() -> None:
        try:
            HomeTab.default_hero_estimator()
        except Exception:
            pass
        try:
            CountryData.get_instance()
        except Exception:
            pass

    threading.Thread(target=_worker, name="seroepi-asset-warmup", daemon=True).start()


def create_app() -> App:
    """Instantiates the SeroEpi PyShiny `App`.

    Returns:
        Configured `shiny.App` instance combining `app_ui()` and `app_server`.

    Examples:
        ```python
        from seroepi.app.app import create_app

        shiny_app = create_app()
        ```
    """
    _warm_reference_assets()
    return App(app_ui(), app_server)


app = create_app()
