# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import sphinx_rtd_theme

# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "zae-engine"
copyright = "2024, zae-park"
author = "zae-park"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []
extensions = ["sphinx.ext.autodoc", "rtds_action"]

autodoc_inherit_docstrings = False

templates_path = ["_templates"]
exclude_patterns = []

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
# TODO: fix toctree (https://devbruce.github.io/etc/etc-03-sphinx_apidoc,theme__to_new/)
master_doc = "index"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Taken from docs.readthedocs.io:
# on_rtd is whether we are on readthedocs.io
on_rtd = os.environ.get("READTHEDOCS", None) == "True"


# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ["_static"]

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "./_static/zaevicon.ico"

# The name of your GitHub repository
rtds_action_github_repo = "zae-park/zae-engine"

# The path where the artifact should be extracted
# Note: this is relative to the conf.py file!
rtds_action_path = "../../zae_engine"

# The "prefix" used in the `upload-artifact` step of the action
rtds_action_artifact_prefix = ""

# A GitHub personal access token is required, more info below
rtds_action_github_token = os.environ["GITHUB_TOKEN"]


# coverage_ignore_functions = [
#     # torch
#     "data_pipeline.resource",
#     "data_pipeline.sample",
#     ]
