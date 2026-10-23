import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from cassetter import Cassette, Cassetter
from shiny.pytest import create_app_fixture

from seroepi.aggregate import Aggregator
from seroepi.country import CountryData
from seroepi.dataset import Dataset
from seroepi.dist import GenomicDistances
from seroepi.estimators.hierarchical import HierarchicalSVIEstimator
from seroepi.estimators.unpooled import UnpooledEstimator
from seroepi.variables import Spatial

shiny_app = create_app_fixture(
    Path(__file__).parent.parent / "src" / "seroepi" / "app" / "app.py",
    scope="session",
)

CASSETTES_DIR = Path(__file__).parent / "cassettes"

vcr = Cassetter(
    cassette_library_dir=CASSETTES_DIR,
    record_mode="none",
)


@pytest.fixture
def cassette(request: pytest.FixtureRequest) -> Iterator[Cassette]:
    """Zero-setup cassetter fixture targeting deterministic network recordings."""
    safe_name = re.sub(r"[^\w\-.]", "_", request.node.name)
    with vcr.use_cassette(f"{safe_name}.yaml", record_mode="once") as active_cassette:
        yield active_cassette


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Returns the tests/data directory path."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def country_data() -> CountryData:
    """Singleton CountryData loaded from countries.joblib."""
    return CountryData.load()


@pytest.fixture(scope="session")
def klebnet_dataset() -> Dataset:
    """Benchmark Klebnet neonatal sepsis dataset with distances attached."""
    return Dataset.example(with_distances=True)


@pytest.fixture(scope="session")
def klebnet_distances(klebnet_dataset: Dataset) -> GenomicDistances:
    """Parsed GenomicDistances from the benchmark 981x981 difference matrix."""
    assert klebnet_dataset.distances is not None
    return klebnet_dataset.distances


@pytest.fixture(scope="session")
def klebnet_aggregator(klebnet_dataset: Dataset) -> Aggregator:
    """Canonical Capsule-by-Country padded Aggregator fitted on the KlebNET example dataset."""
    k_var = klebnet_dataset.var("K_locus")
    country_var = Spatial("Country", title="Country", resolution=Spatial.Resolution.COUNTRY)
    return Aggregator(target=k_var, strata=country_var, pad_zeros=True).fit(klebnet_dataset)


@pytest.fixture(scope="session")
def klebnet_unpooled(klebnet_aggregator: Aggregator) -> UnpooledEstimator:
    """Session-cached Wilson UnpooledEstimator fitted on klebnet_aggregator."""
    return UnpooledEstimator(method="wilson").fit(klebnet_aggregator)


@pytest.fixture(scope="session")
def klebnet_svi(klebnet_aggregator: Aggregator) -> HierarchicalSVIEstimator:
    """Session-cached HierarchicalSVIEstimator fitted on klebnet_aggregator."""
    return HierarchicalSVIEstimator(num_steps=50, num_samples=50).fit(klebnet_aggregator)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Treat 'no tests collected' (exit code 5) as success for pre-commit / testmon runs."""
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
