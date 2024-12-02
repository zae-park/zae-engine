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
print(f"GA_MEASUREMENT_ID: {google_analytics}")
html_context = {"ga_measurement_id": google_analytics}


def setup(app):
    app.add_html_theme("sphinx_rtd_theme", "sphinx_rtd_theme")
    # app.add_css_file("custom.css")  # 필요한 경우 추가 CSS 파일

    # if google_analytics:
    #     # 외부 GA 스크립트 추가 (async 설정)
    #     app.add_js_file(
    #         f"https://www.googletagmanager.com/gtag/js?id={google_analytics}",
    #         async_=True,  # 'async'는 'async_'로 지정해야 합니다.
    #     )
    #     # 인라인 GA 설정 스크립트 추가 (script 태그 제외)
    #     app.add_js_file(
    #         None,
    #         body=f"""
    #             window.dataLayer = window.dataLayer || [];
    #             function gtag(){{dataLayer.push(arguments);}}
    #             gtag('js', new Date());
    #
    #             gtag('config', '{google_analytics}');
    #             """,
    #     )
    # else:
    #     pass
