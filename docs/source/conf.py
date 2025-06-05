# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# import os
# import sys
# sys.path.insert(0, os.path.abspath("../sphedron"))  # Adjust path to your package

import os
import sys
sys.path.insert(0, os.path.abspath('../../sphedron/'))
print("PATH:",os.path.abspath('../../sphedron/'))

project = "Sphedron"
author = "Ayoub Ghriss"
# copyright = "2025, Ayoub Ghriss"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Parses Google/NumPy docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]
autosummary_generate = True
html_theme = "sphinx_rtd_theme"
napoleon_google_docstring = True  # Enable Google docstring parsing
napoleon_numpy_docstring = False  # Disable NumPy docstrings

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store", "tests/*"]
# html_static_path = ["_static"]
