# Configuration file for the Sphinx documentation builder.

import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../zae_engine"))

# -- Project information -----------------------------------------------------
project = "zae-engine"
copyright = "2024, zae-park"
author = "zae-park"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = ".rst"

master_doc = "index"

# -- Options for HTML output -------------------------------------------------
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ["_static"]

html_favicon = "./_static/zaevicon.ico"

# Google Analytics 설정
google_analytics = os.getenv("GA_MEASUREMENT_ID")


def setup(app):
    app.add_html_theme("sphinx_rtd_theme", "sphinx_rtd_theme")
    app.add_css_file("custom.css")  # 필요한 경우 추가 CSS 파일

    if google_analytics:
        app.add_js_file(
            None,
            body=f"""
            <!-- Google Analytics -->
            <script async src="https://www.googletagmanager.com/gtag/js?id={google_analytics}"></script>
            <script>
              window.dataLayer = window.dataLayer || [];
              function gtag(){{dataLayer.push(arguments);}}
              gtag('js', new Date());

              gtag('config', '{google_analytics}');
            </script>
            <!-- End Google Analytics -->
        """,
        )
