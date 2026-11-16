"""Tests for the pieces of the generation baselines that need neither a GPU nor the KG."""
from vakg.bm25 import BM25, tokenize
from vakg.prompts import extract_code, messages

START = "import numpy as np\ndef f(a: np.ndarray) -> np.ndarray:\n    return "
ROW = {"library": "numpy", "version": "1.25.0", "python_version": "3.10", "additional_dependencies": "",
       "problem": "Round the array.", "starting_code": START}


def test_tokenize_keeps_identifiers_whole_and_split():
    assert tokenize("nx.community.greedy_modularity_communities(G)") == [
        "nx", "community", "greedy_modularity_communities", "greedy", "modularity", "communities"]
    assert tokenize("the DataFrame is returned") == ["dataframe", "data", "frame", "returned"]


def test_bm25_prefers_the_document_with_the_rare_query_term():
    docs = [tokenize(t) for t in ("numpy.around round an array to the given decimals",
                                  "numpy.sum sum of array elements over an axis",
                                  "numpy.prod product of array elements over an axis")]
    ranked = BM25(docs).top(tokenize("round the array"), k=2)
    assert ranked[0][0] == 0 and ranked[0][1] > ranked[1][1]
    assert BM25(docs).top(tokenize("nothing matches zzz"), k=3) == []


def test_extract_fenced_answer():
    answer = "Here is the code:\n```python\nimport numpy as np\ndef f(a):\n    return np.round(a)\n```\nDone."
    assert extract_code(answer, START) == "import numpy as np\ndef f(a):\n    return np.round(a)\n"


def test_extract_puts_starting_code_back_when_only_the_continuation_is_returned():
    assert extract_code("```python\nnp.round(a)\n```", START) == START + "np.round(a)\n"


def test_extract_unterminated_fence_and_no_fence():
    assert extract_code("```python\ndef f(a):\n    return a", START) == "def f(a):\n    return a\n"
    assert extract_code("def f(a):\n    return a", START) == "def f(a):\n    return a\n"


def test_b0_and_b1_prompts_differ_only_by_the_documentation():
    doc = {"path": "numpy.around", "signature": "(a, decimals=0)", "doc": "Evenly round."}
    b0, b1 = messages(ROW)[1]["content"], messages(ROW, [doc])[1]["content"]
    assert "numpy.around(a, decimals=0)" in b1 and "Documentation" not in b0
    assert b1.replace("Documentation retrieved from numpy 1.25.0 (it may or may not be relevant):\n\n"
                      "### numpy.around(a, decimals=0)\nEvenly round.\n\n", "") == b0
