"""Prompts for the generation baselines and extraction of the code from an answer.

B0 (LLM alone) and B1 (BM25-retrieved documentation) share the same prompt; the
only difference is the documentation block, so any gap between them is due to it.
"""
from __future__ import annotations

import re

SYSTEM = ("You are an expert Python programmer. "
          "Answer with a single Python code block and no explanation.")

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)
_OPEN_FENCE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\n(.*)\Z", re.S)
_DEFINED = re.compile(r"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z_]\w*)", re.M)


def user_prompt(row: dict, docs: list[dict] | None = None) -> str:
    lib, version = row["library"], row["version"]
    env = f"{lib}=={version}, Python {row['python_version']}"
    if row.get("additional_dependencies"):
        env += f", also installed: {row['additional_dependencies']}"
    parts = [f"Environment: {env}", ""]
    if docs:
        parts.append(f"Documentation retrieved from {lib} {version} (it may or may not be relevant):")
        for d in docs:
            parts += ["", f"### {d['path']}{d.get('signature', '')}", d["doc"]]
        parts.append("")
    parts += [
        f"Task: {row['problem'].strip()}",
        "",
        f"Complete the starting code so that it solves the task, using only APIs that exist in "
        f"{lib} {version}. Return the complete code (starting code included) in one ```python block.",
        "",
        "Starting code:",
        "```python",
        row["starting_code"].rstrip("\n"),
        "```",
    ]
    return "\n".join(parts)


def messages(row: dict, docs: list[dict] | None = None) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(row, docs)}]


def extract_code(answer: str, starting_code: str = "") -> str:
    """The program contained in a model answer.

    Takes the longest fenced block (or an unterminated one, if generation was cut
    off; or the whole answer when there is no fence). If the answer does not define
    what the starting code defines, the model returned only the continuation, so the
    starting code is put back in front of it.
    """
    blocks = [body for lang, body in _FENCE.findall(answer) if lang.lower() in ("", "python", "py", "python3")]
    if blocks:
        code = max(blocks, key=len)
    else:
        open_block = _OPEN_FENCE.search(answer)
        code = open_block.group(1) if open_block else answer
    code = code.strip("\n") + "\n"
    expected = set(_DEFINED.findall(starting_code))
    if expected and not expected & set(_DEFINED.findall(code)):
        code = starting_code + code
    return code
