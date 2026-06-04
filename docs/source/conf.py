# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from importlib.metadata import version as _pkg_version, PackageNotFoundError

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "pyCSA"
copyright = "2024, Ray Chew"
author = "Ray Chew"

# The full version, including alpha/beta/rc tags. Resolved dynamically from the
# installed package metadata so the docs never drift from pyproject.toml; falls
# back to the in-tree __version__, then a last-resort default.
try:
    release = _pkg_version("pyCSA")
except PackageNotFoundError:
    try:
        from pycsa import __version__ as release
    except Exception:
        release = "0.0.0"
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_changelog",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc / autosummary ---------------------------------------------------

# Generate the per-module/-class stub pages under _autosummary/ at build time
# so the API reference always mirrors the live ``pycsa`` package.
autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# All runtime dependencies (Cartopy, numba, netCDF4, h5py, pandas, ...) are
# pinned in requirements.txt and installed on the docs runner (CI also adds
# libgeos-dev for Cartopy), so autodoc imports the real packages and documents
# real signatures. No mocking — mocking numba in particular would replace the
# ``@nb.njit`` decorator with a stub and mangle the decorated callables.


def _skip_reexports(app, what, name, obj, skip, options):
    """Document each object once, in its defining module.

    The ``pycsa.core.var`` deprecation shim and the ``pycsa.data`` /
    ``pycsa.compute`` package ``__init__`` re-export classes via ``__all__``.
    Without this, autodoc documents them on both the re-exporting page and the
    defining page, producing "duplicate object description" and ambiguous
    cross-reference warnings. Skip a member whose ``__module__`` differs from
    the module currently being documented.
    """
    if skip or what != "module":
        return None
    defined_in = getattr(obj, "__module__", None)
    if not defined_in or not defined_in.startswith("pycsa"):
        return None
    # The generated stub's docname is e.g. "_autosummary/pycsa.data"; the last
    # path segment is the dotted name of the module being documented.
    current = app.env.docname.rsplit("/", 1)[-1]
    if current.startswith("pycsa") and defined_in != current:
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", _skip_reexports)

# -- napoleon (NumPy-style docstrings) ---------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_context = {
    # ...
    "default_mode": "light"
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
numfig = True


# -- GraphViz configuration ----------------------------------
graphviz_output_format = "svg"
