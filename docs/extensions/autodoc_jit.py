#!/usr/bin/env python3


from __future__ import annotations

import typing as t

from numba.core.dispatcher import Dispatcher

if t.TYPE_CHECKING:
    from sphinx.application import Sphinx


def process_jit_functions(
    app: Sphinx, what: str, name: str, obj: t.Any, options, lines: list[str]
):
    if what != "function" or not isinstance(obj, Dispatcher):
        return

    indent = 4 * " "
    new_lines = []

    overloads = obj.overloads.values()
    options = dict(obj.targetoptions)

    new_lines += ["", ".. admonition:: Compiled with Numba", ""]
    if options:
        options_str = ", ".join(
            [f"{k}: *{v}*" for k, v in options.items() if v is not None]
        )
    new_lines += [indent + f"* **Options:** {options_str}"]

    if overloads:
        new_lines += [indent + "* **Signatures:**"]
        for cres in overloads:
            new_lines += [2 * indent + "* " + str(cres.signature)]

    new_lines += [""]

    for line in reversed(new_lines):
        lines.insert(0, line)


def setup(app: Sphinx):  # noqa: D103
    app.setup_extension("sphinx.ext.autodoc")

    app.connect("autodoc-process-docstring", process_jit_functions)

    return dict(
        version="0.1",
        parallel_read_safe=True,
        parallel_write_safe=True,
    )
