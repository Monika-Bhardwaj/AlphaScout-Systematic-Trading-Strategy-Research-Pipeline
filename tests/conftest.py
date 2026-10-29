import warnings

import pytest

warnings.filterwarnings("ignore", message="Sortedness of columns")

from alphascout.data.pit_store import PITStore
from alphascout.data.synthetic import generate_synthetic


@pytest.fixture(scope="session")
def raw():
    return generate_synthetic(60, 900, seed=3, planted_pairs=3, planted_momentum=1.0)


@pytest.fixture(scope="session")
def store(raw):
    return PITStore(raw)


@pytest.fixture(scope="session")
def panel(store):
    return store.build_panel()
