# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# import os
# import sys
# sys.path.insert(0, os.path.abspath("../sphedron"))  # Adjust path to your package

# import os
# import sys
# sys.path.insert(0, os.path.abspath('../../sphedron/'))
# print("PATH:",os.path.abspath('../../sphedron/'))

import sphedron
import inspect
import importlib

project = "Sphedron"
author = "Ayoub Ghriss"
copyright = "2025, Ayoub Ghriss"

# add_module_names = False
# typehints_fully_qualified = False  # sphinx-autodoc-typehints >= 2.0


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # 'sphinx_autodoc_typehints',
    # 'sphinx_copybutton',
    # 'nbsphinx',
    # 'pyg',
]

autosummary_generate = True
autosummary_generate_overwrite = True
html_theme = "sphinx_rtd_theme"
napoleon_google_docstring = True  # Enable Google docstring parsing
napoleon_numpy_docstring = True  # Disable NumPy docstrings
# suppress_warnings = ["autodoc.import_object"]
# autodoc_default_flags = {
#     # "members": True,
#     # "show-inheritance": True,
# "member-order": "groupwise",  # groups methods vs attributes/properties
#     # "undoc-members": False,
# "private-members": False,
# }


templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store", "tests/*"]
html_static_path = ["_static"]


def rst_jinja_render(app, _, source):
    if hasattr(app.builder, "templates"):
        rst_context = {"sphedron": sphedron}
        source[0] = app.builder.templates.render_string(source[0], rst_context)


def autodoc_skip_member(app, what, name, obj, skip, options):
    # print(name)
    # if name in {"__init__", "__call__"}:
    # return False
    # Skip single-underscore private and non-magic double-underscore
    if name.startswith("_") and not (
        name.startswith("__") and name.endswith("__")
    ):
        return True
    return skip

    # def setup(app):
    # app.connect("source-read", rst_jinja_render)
    # app.connect("autodoc-skip-member", autodoc_skip_member)
    # app.connect("autodoc-skip-attribute", autodoc_skip_member)

    # app.add_js_file('js/version_alert.js')

    # Do not drop type hints in signatures:
    # del app.events.listeners["autodoc-process-signature"]

    # conf.py
    # def _split_methods(fullname, names):
    #     import importlib, inspect
    #     mod, clsname = fullname.rsplit(".", 1)
    #     cls = getattr(importlib.import_module(mod), clsname)
    #     out = {"classmethods": [], "staticmethods": [], "instancemethods": []}
    #     for n in names:
    #         try:
    #             attr = inspect.getattr_static(cls, n)
    #         except Exception:
    #             continue
    #         if isinstance(attr, classmethod):
    #             out["classmethods"].append(n)
    #         elif isinstance(attr, staticmethod):
    #             out["staticmethods"].append(n)
    #         elif inspect.isfunction(attr):
    #             out["instancemethods"].append(n)
    #     return out

    # def _split_attributes(fullname, names):
    #     import importlib, inspect
    #     mod, clsname = fullname.rsplit(".", 1)
    #     cls = getattr(importlib.import_module(mod), clsname)
    #     props, data = [], []
    #     for n in names:
    #         try:
    #             attr = inspect.getattr_static(cls, n)
    #         except Exception:
    #             continue
    #         (props if isinstance(attr, property) else data).append(n)
    #     return {"properties": props, "data": data}

    # def _register_filters(app):
    #     env = app.builder.templates.environment  # official way to extend template env
    #     env.filters["split_methods"] = _split_methods
    #     env.filters["split_attributes"] = _split_attributes

    # def setup(app):
    #     app.connect("builder-inited", _register_filters)  # recommended event


def classify(names, fullname):
    """Split member names into properties vs data, and instance/class/static methods."""
    from sphinx.util import logging

    log = logging.getLogger(__name__)
    mod, clsname = fullname.rsplit(".", 1)
    log.warning(f"Classify print: {fullname, names, mod, clsname}")
    cls = getattr(importlib.import_module(mod), clsname)

    props, data = [], []
    inst_meths, cls_meths, static_meths = [], [], []

    for n in names or []:
        try:
            a = inspect.getattr_static(cls, n)
        except Exception:
            continue
        if isinstance(a, property):
            props.append(n)
        elif isinstance(a, classmethod):
            cls_meths.append(n)
        elif isinstance(a, staticmethod):
            static_meths.append(n)
        elif inspect.isfunction(a):
            inst_meths.append(n)
        else:
            data.append(n)

    return {
        "properties": props,
        "data": data,
        "methods": inst_meths,
        "classmethods": cls_meths,
        "staticmethods": static_meths,
    }


def _register_filters(app):
    env = app.builder.templates.environment
    env.filters["classify"] = classify


def setup(app):
    app.connect("builder-inited", _register_filters)
    app.connect("source-read", rst_jinja_render)
    app.connect("autodoc-skip-member", autodoc_skip_member)
