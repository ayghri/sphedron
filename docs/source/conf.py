# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

project = "Sphedron"
author = "Ayoub Ghriss"
copyright = "2025, Ayoub Ghriss"

# -- Extensions --------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

# -- Autodoc / Napoleon -------------------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- HTML output --------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"

html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "show_nav_level": 0,
    "navigation_depth": 4,
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo.svg",
    },
}

html_sidebars = {
    "**": ["sidebar-nav-bs"],
}

html_css_files = ["custom-style.css"]
html_js_files = ["collapse-code.js"]
html_static_path = ["_static"]

# -- Notebook rendering (nbsphinx) -------------------------------------------

nbsphinx_execute = os.environ.get("SPHEDRON_DOCS_EXECUTE_NOTEBOOKS", "never")
nbsphinx_timeout = int(os.environ.get("SPHEDRON_DOCS_NOTEBOOK_TIMEOUT", "300"))
nbsphinx_allow_errors = False
nbsphinx_widgets_path = ""  # suppress ipywidgets warning; widgets not used

suppress_warnings = ["nbsphinx.ipywidgets"]
