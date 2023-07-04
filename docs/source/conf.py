# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
import sphinx_press_theme

project_path = '../../SCTrack'
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
# sys.path.insert(0, os.path.abspath(project_path))

apidoc_module_dir = project_path
apidoc_output_dir = 'python_apis'
# apidoc_excluded_paths = ['tests']
apidoc_separate_modules = True

project = 'SC-Track'
copyright = '2023, Li Chengxin'
author = 'Li Chengxin'
release = '0.0.1 alpha'

master_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinxcontrib.apidoc',
    'sphinx.ext.viewcode',
    'm2r2'
]

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'


html_theme = "press"

html_static_path = ['_static']
