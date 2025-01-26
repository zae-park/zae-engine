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
    "sphinx.ext.autodoc",  # 자동 문서 생성
    "sphinx.ext.coverage",  # 문서화 커버리지
    "sphinx.ext.napoleon",  # google-style, numpydoc-style 지원
    "sphinx.ext.githubpages",  # 깃헙 페이지 연동
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",  # 주석 내 예시 코드 실행
    "sphinx.ext.mathjax",  # 수식 표현 지원
    # "sphinxcontrib.gtagjs"  # google 태그 관리자 분석 추가
]

templates_path = ["_templates"]
locale_dirs = ["source/locale/"]  # 번역 파일이 저장될 디렉터리
gettext_compact = False  # 번역 파일 분리 생성
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
