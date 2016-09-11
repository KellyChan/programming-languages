###################
Sphinx-Framework
###################

Sphinx-Frame is a documentation project template which is built by python-sphinx.

****************************
Folder Structure
****************************


Expected project structure

::

    ./sphinx-framework/                             # the project root
          |---- sphinx-framework/                   # the source codes
          |       | 
          |       |----- projects/                  # write the docs here
          |       |----- requirements/              # packages needed
          |       |
          |       |----- develop.sh                 # run the html server
          |       |----- run_tests.sh               # test the html pages
          |       |
          |       |----- docs/                      # developer docs
          |       |----- RELEASE.rst                # release notes
          |       |----- AUTHORS                    # author status
          |       |----- LICENSE.rst                # license
          |       |----- README.rst                 # read me
          |
          |---- venvs/                              # the project env
                  |----- sphinx-framework/          # python virtualenv


****************************
Installing the devstack
****************************

build a project environment

::

    $ cd /path/to/project
    $ mkdir sphinx-framework
    $ cd sphinx-framework
    $ mkdir venv
    $ virtualenv venvs/sphinx-framework
    $ source venvs/sphinx-framework/bin/activate

clone the repo

::

    $ cd /path/to/sphinx-framework
    $ git clone http://github.com/KellyChan/sphinx-framework.git
    $ cd sphinx-framework

install the packages


::

    $ cd sphinx-framework
    $ pip install -r requirements/travis_requirements.txt
    $ pip install -r requirements/tools.txt


*******************************
Writing the documentations
*******************************

rename the project

::

    $ cd sphinx-framework
    $ cp projects project_name


the static pages are created by `RST`_ files and `Sphinx`_, please check the references for more details.

.. _RST: http://docutils.sourceforge.net/rst.html
.. _Sphinx: http://sphinx-doc.org

********************************
Run the project
********************************

run a specific project

::

    $ ./develop.sh projects/project_name

open the http://localhost:9090 in a browser to check the site.


run tests

::

    $ ./run_tests.sh                           # run all the projects set in run_test.sh
    $ ./run_tests.sh projects/project_name     # run a specific project

It will generate the test report in the test_root.

******************************
Custom Theme
******************************

install a new theme

::

    $ cd sphinx-framework/projects/_themes
    $ git clone http://custome_theme
    $ cd sphinx-framework/projects/project_name/source
    $ vim conf.py                                       # update the theme_name here

conf.py: update the theme

::

    # The theme to use for HTML and HTML Help pages.  See the documentation for
    # a list of builtin themes.
    # html_theme = 'default'
    html_theme = 'custom_theme'

    html_favicon = '../../_themes/custom_theme/static/css/favicon.ico'

********************************
LICENSE
********************************

It is followed the `MIT LICENSE`_.

.. _`MIT LICENSE`: LICENSE

