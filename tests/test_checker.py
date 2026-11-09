"""Regression tests for the checker (need the KG: run vakg.envs and vakg.kg first)."""
from pathlib import Path

import pytest

from vakg.checker import check_code

KG = Path(__file__).resolve().parents[1] / "kg"
pytestmark = pytest.mark.skipif(not (KG / "numpy.nt").exists(), reason="KG not built")


def symbols(code, lib, version, severity="error"):
    return {i.symbol for i in check_code(code, lib, version) if i.severity == severity}


def test_hallucinated_keyword_is_an_error():
    code = "import numpy as np\nnp.convolve([1], [1], mode='full', method='fft')\n"
    assert symbols(code, "numpy", "1.25.0") == {"numpy.convolve(method=)"}


def test_method_on_annotated_variable_follows_the_class():
    code = "import numpy as np\ndef f(a: np.ndarray):\n    return a.sum(axis=0), a.cumprod()\n"
    assert symbols(code, "numpy", "1.25.0") == set()


def test_reexported_submodule_alias():
    # nx.community is networkx.algorithms.community re-exported (vakg:aliasOf)
    code = "import networkx as nx\nnx.community.greedy_modularity_communities\n"
    assert symbols(code, "networkx", "2.8") == set()


def test_removed_member_is_reported_with_history():
    # DataFrame.append was removed in pandas 2.0
    code = "import pandas as pd\ndf = pd.DataFrame()\ndf.append({'a': 1}, ignore_index=True)\n"
    issues = [i for i in check_code(code, "pandas", "2") if i.severity == "error"]
    assert [i.symbol for i in issues] == ["pandas.DataFrame.append"]
    assert "1.5.1" in issues[0].exists_in
    assert symbols(code, "pandas", "1.5.1") == set()


def test_private_names_are_not_judged():
    code = "import networkx as nx\nnx.community._naive_greedy_modularity_communities\n"
    assert symbols(code, "networkx", "2.4") == set()
