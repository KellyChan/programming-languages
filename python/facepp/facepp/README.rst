##########################################
Facepp
##########################################

-----------------------------
How to run
-----------------------------

::

    $ cp settings/local.sample.py settings/local.py
    $ vim settings.local.py                 # update API_KEY and API_SECRET
    
    $ python run.py


-----------------------------
How to add a feature
-----------------------------

::

    $ cp apps/app.sample.py feature_name.py
    $ vim feature_name.py                   # write your awesome feature
    $ vim run.py                            # call your new feature
