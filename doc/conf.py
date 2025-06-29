# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import fronts_toolbox

## Project information

project = "fronts-toolbox"
copyright = "2025, Clément Haëck"
author = "Clément Haëck"
version = fronts_toolbox.__version__
release = fronts_toolbox.__version__

print(f"{project}: {version}")

## General configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

add_module_names = False
toc_object_entries_show_parents = "hide"

pygments_style = "default"

## Autodoc config
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "all"
autodoc_class_content = "both"
autodoc_class_signature = "mixed"

python_use_unqualified_type_names = True

autodoc_default_options = {"show-inheritance": True, "inherited-members": False}

## Autosummary config
autosummary_generate = ["api.rst"]

## Napoleon config
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_type = False
# napoleon_type_aliases = autodoc_type_aliases.copy()

## Intersphinx config
intersphinx_mapping = {
    "dask": ("https://docs.dask.org/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

## Options for HTML output

github = "https://github.com/Descanonge/fronts-toolbox"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = project
html_theme_options = dict(
    # Social icons
    icon_links=[
        dict(name="Repository", url=github, icon="fa-brands fa-square-gitlab"),
    ],
    github_url=github,
    # TOC
    secondary_sidebar_items=["page-toc"],
    show_toc_level=2,
    collapse_navigation=False,
    # Navigation bar
    navbar_start=["navbar-logo", "navbar-icon-links"],
    navbar_center=["navbar-nav"],
    navbar_end=["search-button"],
    # Footer
    show_prev_next=False,
    article_footer_items=[],
    content_footer_items=[],
    footer_start=["copyright", "last-updated"],
    footer_end=["sphinx-version", "theme-version"],
)

html_last_updated_fmt = "%Y-%m-%d"
