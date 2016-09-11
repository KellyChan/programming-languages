# -*- coding: utf-8 -*-
#

import sys, os

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

master_doc = 'index'

# The suffix of source filenames.
source_suffix = '.rst'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'



# Add any paths that contain templates here, relative to this directory.
#templates_path.append('source/_templates')

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path.append('source/_static')

# The short X.Y version.
version = ''
# The full version, including alpha/beta/rc tags.
release = ''

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = 'default'
html_theme = 'custom_theme'

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []
html_theme_path = ['../../_themes']

html_favicon = '../../_themes/custom_theme/static/css/favicon.ico'

#html_use_smartypants = True
html_use_smartypants = True

if on_rtd:
    html_context = {
       "on_rtd" : on_rtd,
       "google_analytics_id" : '',
       "disqus_shortname" : '',
       "github_base_account" : 'KellyChan',
       "github_project" : 'sphinx-framework',
    }

# General information about the project.
project = u'Sphinx Framework'
copyright = u'2015, ATR'

# The short X.Y version.
version = ''
# The full version, including alpha/beta/rc tags.
release = ''

#remove directory when content is first added to it, and add to index
exclude_patterns = ['links.rst', 'reusable/*']
