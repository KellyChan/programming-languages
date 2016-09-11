####################################
FacePlusPlus Python SDK
####################################

This is the Face++ python SDK suite. Note that python2.7 is required.

----------------------
API
----------------------

``facepp.py``: This is the underlying API implementation.

----------------------
API Calls
----------------------

CLI: cmdtool.py
~~~~~~~~~~~~~~~~~~

This is an interactive command line tool which could be used to experiment
with Face++ APIs. It is recommended to have ipython installed so that you can
have tab-completion and some other nice features.

- ``apikey.cfg``: `cp apikey.sample.cfg apikey.cfg`, input `API_KEY` and `API_SECRET`
- ``cmdtool.py``: type `python cmdtool.py` to start a python shell 

Face++ Cli:

::

    $ api.detection.detect(img = File(r'<path to the image file>'))

    $ api.detection.detect(url = 'http://faceplusplus.com/2014/01/2.jpg')
    $ api.detection.detect(img = File(r'/tmp/test.jpg'))
    $ api.person.create(person_name = 'Bob')
    $ api.info.get_group_list()


For more features, please refer to the official document `Application management system command line tool`_.

.. _`Application management system command line tool`: http://www.faceplusplus.com/application-management-system-command-line-tool/


Scripting: hello.py
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a comprehensive demo for Face++ APIs. See the comments in the source
code for details.

