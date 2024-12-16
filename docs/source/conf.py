# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'NN_Profiles'
copyright = '2024, Isaac Malsky'
author = 'Isaac Malsky'
release = '0.1'

# -- General configuration ---------------------------------------------------
extensions = []
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# Ensure correct base URL for static assets
html_baseurl = "https://imalsky.github.io/NN_Profiles/"
