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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'scalfmm3'
copyright = '2020, Pierre Esterie'
author = 'Pierre Esterie'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

## Add any Sphinx extension module names here, as strings. They can be
## extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
## ones.
#extensions = ["breathe"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

## List of patterns, relative to source directory, that match files and
## directories to ignore when looking for source files.
## This pattern also affects html_static_path and html_extra_path.
#exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
#
#
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

## Add any paths that contain custom static files (such as style sheets) here,
## relative to this directory. They are copied after the builtin static files,
## so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']
#
#breathe_default_project = "scalfmm3"

# The `extensions` list should already be in here from `sphinx-quickstart`
extensions = [
    # there may be others here already, e.g. 'sphinx.ext.mathjax'
    'breathe',
    'exhale',
    'recommonmark'
]

# Setup the breathe extension
breathe_projects = {
    "scalfmm3": "./docs/doxygen/xml"
}
breathe_default_project = "scalfmm3"

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder":     "./api",
    "rootFileName":          "library_root.rst",
    "doxygenStripFromPath":  "..",
    # Heavily encouraged optional argument (see docs)
    "rootFileTitle":         "Library API",
    # Suggested optional arguments
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":    "INPUT = ../include"
}

# Setup the exhale extension
exhale_args_tools = {
    # These arguments are required
    "containmentFolder":     "./tools",
    "rootFileName":          "tools_root.rst",
    "doxygenStripFromPath":  "..",
    # Heavily encouraged optional argument (see docs)
    "rootFileTitle":         "Tools codes",
    # Suggested optional arguments
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":    "INPUT = ../tools"
}
# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'

import pathlib

# The readme and quickstart that already exist.
readme_path = pathlib.Path(__file__).parent.resolve().parent / "README.md"
quickstart_path = pathlib.Path(__file__).parent.resolve().parent / "QUICKSTART.md"
# We copy a modified version here
readme_target = pathlib.Path(__file__).parent / "readme.md"
quickstart_target = pathlib.Path(__file__).parent / "quickstart.md"

def dump_md(path, target):
    with target.open("w") as outf:
        # Change the title to "Readme"
        lines = []
        for line in path.read_text().split("\n"):
            if line.startswith("# "):
                # Skip title, because we now use "Readme"
                # Could also simply exclude first line for the same effect
                continue
            lines.append(line)
        outf.write("\n".join(lines))

dump_md(readme_path, readme_target)
dump_md(quickstart_path, quickstart_target)

