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
sys.path.insert(0, os.path.abspath("../nn"))
sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = 'proj2'
copyright = '2021, Jodok Vieli, Puck van Gerwen, Felix Hoppe'
author = 'Jodok Vieli, Puck van Gerwen, Felix Hoppe'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["../test.py", "test.py", "../test/", "test/"] 

# latex 
#latex_engine = "pdflatex"
latex_documents = [('index', 'proj2.tex', u'Proj2', u'PuckVG', 'manual'),]
