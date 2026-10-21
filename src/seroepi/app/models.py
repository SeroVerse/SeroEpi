"""SeroEpi domain `SessionState` subclassing `shiny_kit.state.BaseSessionState`."""

from typing import Any, Literal

import polars as pl
from shiny import reactive
from shiny_kit.decorators import reactive_prop
from shiny_kit.state import BaseSessionState, artifact_registry

from seroepi.aggregate import Aggregator
from seroepi.app.caching import ModelCache, default_cache
from seroepi.dataset import Dataset
from seroepi.estimators.base import BaseEstimator
from seroepi.estimators.formulation import Formulation
from seroepi.variables import Target, VariableFilter

type RegistryKind = Literal["datasets", "targets", "aggregators", "estimators", "formulations"]


class SessionState(BaseSessionState):
    """SeroEpi session state container holding reactive pipeline registries, active filters, and model cache.

    Subclasses [`shiny_kit.state.BaseSessionState`](file:///Users/tsta0015/Programming/shiny-kit/src/shiny_kit/state.py)
    to declare five reactive [`ArtifactRegistry`](file:///Users/tsta0015/Programming/shiny-kit/src/shiny_kit/state.py)
    containers (`datasets`, `targets`, `aggregators`, `estimators`, `formulations`) alongside
    reactive quality-control (`qc_enabled_map`) and variable filter (`variable_filters`) state.

    Attributes:
        datasets (ArtifactRegistry[Dataset]): Registry of loaded or merged `Dataset` instances.
        targets (ArtifactRegistry[Target]): Registry of engineered antigenic `Target` definitions.
        aggregators (ArtifactRegistry[Aggregator]): Registry of fitted `Aggregator` count tables.
        estimators (ArtifactRegistry[BaseEstimator]): Registry of fitted `BaseEstimator` models.
        formulations (ArtifactRegistry[Formulation]): Registry of fitted `Formulation` designs.
        qc_enabled_map (dict[str, bool]): Reactive mapping from QC column names to enabled flags.
        variable_filters (dict[str, VariableFilter]): Reactive mapping from column names to
            active `VariableFilter` specifications.

    Examples:
        ```python
        from seroepi.app.models import SessionState
        from seroepi.examples import KlebnetNeonatalSepsis

        state = SessionState()
        ds = KlebnetNeonatalSepsis.load()
        state.register_dataset(ds)
        assert state.active_dataset is not None
        ```
    """

    datasets = artifact_registry[Dataset](
        "Datasets",
        icon="database",
        badge_class="bg-primary",
        namer=lambda ds: ds.name,
        toast=lambda ds: f"Loaded dataset '{ds.name}' ({len(ds):,} rows).",
        on_register="_on_dataset_registered",
        on_select="_on_dataset_selected",
        on_remove="_on_dataset_removed",
    )
    targets = artifact_registry[Target](
        "Targets",
        icon="target",
        badge_class="bg-success",
        namer=lambda t: t.name,
    )
    aggregators = artifact_registry[Aggregator](
        "Aggregators",
        icon="layers",
        badge_class="bg-secondary",
        show_in_navbar=False,
        namer=lambda agg: agg.summary_label,
        toast=lambda agg: f"Aggregated '{agg.summary_label}' ({len(agg.data_):,} rows).",
    )
    estimators = artifact_registry[BaseEstimator](
        "Estimators",
        icon="trending-up",
        badge_class="bg-info text-dark",
        namer=lambda est: est.name or f"{est.display_name} ({est.summary_label})",
        toast=lambda est: (
            f"Fitted estimator '{est.display_name}' "
            f"({len(est.estimates_) if est.estimates_ is not None else 0:,} rows)."
        ),
        on_register="_on_estimator_registered",
    )
    formulations = artifact_registry[Formulation](
        "Formulations",
        icon="shield-check",
        badge_class="bg-warning text-dark",
        namer=lambda form: form.name or f"{form.display_name} ({form.summary_label})",
        toast=lambda form: f"Solved formulation '{form.display_name}'.",
    )

    qc_enabled_map: reactive_prop[dict[str, bool]] = reactive_prop({})
    variable_filters: reactive_prop[dict[str, VariableFilter]] = reactive_prop({})

    def __init__(self, *, cache: ModelCache | None = None, navset_id: str = "main_nav") -> None:
        """Initializes a `SessionState` instance with a shared `ModelCache` and top-level navbar ID.

        Args:
            cache: Disk/memory `ModelCache` used to memoize expensive pipeline operations.
                Defaults to `default_cache`.
            navset_id: DOM ID of the top-level `ui.page_navbar` container. Defaults to `"main_nav"`.
        """
        super().__init__(cache=cache or default_cache, navset_id=navset_id)
        self.cache: ModelCache = cache or default_cache
        self._active_dataset_cache: tuple[tuple[Any, ...], Dataset] | None = None

    @property
    def raw_dataset(self) -> Dataset | None:
        """Reactively returns the unfiltered active `Dataset` from `self.datasets`.

        Returns:
            The currently selected `Dataset` prior to applying QC or variable filters,
            or `None` if no dataset is registered.
        """
        return self.datasets.active

    @property
    def active_filter_exprs(self) -> list[pl.Expr]:
        """Reactively collects all enabled QC and `VariableFilter` expressions for `raw_dataset`.

        Returns:
            List of Polars boolean filter expressions to apply to `raw_dataset`.
        """
        ds = self.raw_dataset
        if ds is None:
            return []

        exprs: list[pl.Expr] = []
        qc_map = self.qc_enabled_map
        for qc_var in ds.qc_variables:
            if qc_map.get(qc_var.column, False) and qc_var.column in ds.data.columns:
                exprs.append(qc_var.filter_expr())

        for col, spec in self.variable_filters.items():
            if col in ds.data.columns:
                expr = spec.to_expr()
                if expr is not None:
                    exprs.append(expr)

        return exprs

    @property
    def active_dataset(self) -> Dataset | None:
        """Reactively returns the active `Dataset` with all enabled QC and variable filters applied.

        Memoizes the filtered `Dataset` against the identity of `raw_dataset` and the
        active filter keys so repeated reactive reads within the same state do not
        recompute Polars DataFrame slices.

        Returns:
            The filtered active `Dataset`, or `None` if no dataset is registered.
        """
        ds = self.raw_dataset
        if ds is None:
            self._active_dataset_cache = None
            return None
        exprs = self.active_filter_exprs
        if not exprs:
            return ds
        key = (
            id(ds),
            tuple(sorted((col, bool(on)) for col, on in self.qc_enabled_map.items() if on)),
            tuple((col, spec.cache_key()) for col, spec in sorted(self.variable_filters.items())),
        )
        if self._active_dataset_cache is not None and self._active_dataset_cache[0] == key:
            return self._active_dataset_cache[1]
        filtered = ds.filter(*exprs)
        self._active_dataset_cache = (key, filtered)
        return filtered

    def reset_dataset_filters(self, ds: Dataset | None = None) -> None:
        """Resets QC switches and custom variable filters for `ds` (or `self.raw_dataset`).

        Args:
            ds: Optional target `Dataset` whose QC variables should be initialized to `False`.
                Defaults to `self.raw_dataset`.
        """
        self._active_dataset_cache = None
        with reactive.isolate():
            target_ds = ds if ds is not None else self.raw_dataset
            self.qc_enabled_map = {v.column: False for v in target_ds.qc_variables} if target_ds is not None else {}
            self.variable_filters = {}

    def _on_dataset_registered(self, ds: Dataset, *, reset_filters: bool = True, **_: Any) -> None:
        if reset_filters:
            self.reset_dataset_filters(ds)

    def _on_dataset_selected(self, ds: Dataset | None, *, reset_filters: bool = True, **_: Any) -> None:
        if reset_filters:
            self.reset_dataset_filters(ds)

    def _on_dataset_removed(self, _removed: Dataset, **_: Any) -> None:
        self.reset_dataset_filters()

    def _on_estimator_registered(self, est: BaseEstimator, **_: Any) -> None:
        if est.aggregator_ is not None and self.aggregators.count == 0:
            self.aggregators.register(est.aggregator_)

    def on_reset(self) -> None:
        """Clears cached filtered dataset and variable filter state when `reset_all()` is called."""
        self.reset_dataset_filters(None)

    def register_dataset(self, ds: Dataset, *, name: str | None = None, reset_filters: bool = True) -> Dataset:
        """Registers `ds` as the active dataset and optionally resets filter state.

        Args:
            ds: `Dataset` instance to register.
            name: Optional explicit registry key. Defaults to `ds.name`.
            reset_filters: Whether to clear active QC and variable filters upon registration.

        Returns:
            The registered `Dataset` instance.
        """
        return self.datasets.register(ds, name=name, set_active=True, reset_filters=reset_filters)

    @property
    def active_aggregator(self) -> Aggregator | None:
        """Reactively returns the currently active fitted `Aggregator`.

        Returns:
            The active `Aggregator` instance, or `None` if none is registered.
        """
        return self.aggregators.active

    def register_aggregator(self, agg: Aggregator, *, name: str | None = None) -> Aggregator:
        """Registers `agg` as the active aggregator using its summary label if no explicit name is given.

        Args:
            agg: Fitted `Aggregator` instance to register.
            name: Optional explicit registry key. Defaults to `agg.summary_label`.

        Returns:
            The registered `Aggregator` instance.
        """
        return self.aggregators.register(agg, name=name, set_active=True)

    @property
    def active_estimator(self) -> BaseEstimator | None:
        """Reactively returns the currently active fitted `BaseEstimator`.

        Returns:
            The active `BaseEstimator` instance, or `None` if none is registered.
        """
        return self.estimators.active

    def register_estimator(self, est: BaseEstimator, *, name: str | None = None) -> BaseEstimator:
        """Registers `est` as the active estimator using a descriptive registry key.

        Args:
            est: Fitted `BaseEstimator` instance to register.
            name: Optional explicit registry key.

        Returns:
            The registered `BaseEstimator` instance.
        """
        return self.estimators.register(est, name=name, set_active=True)

    @property
    def active_formulation(self) -> Formulation | None:
        """Reactively returns the currently active fitted `Formulation`.

        Returns:
            The active `Formulation` instance, or `None` if none is registered.
        """
        return self.formulations.active

    def register_formulation(self, form: Formulation, *, name: str | None = None) -> Formulation:
        """Registers `form` as the active formulation using a descriptive registry key.

        Args:
            form: Fitted `Formulation` instance to register.
            name: Optional explicit registry key.

        Returns:
            The registered `Formulation` instance.
        """
        return self.formulations.register(form, name=name, set_active=True)
