#########################################
Time and Clock
#########################################


Question
------------------------

Given a time, calculate the angle between the hour and minute hands.


Algorithm
------------------------

variable

- time
  - hour
  - minute

formula 

::

    angle between hands = (angle of hour - angle of minute) % 360
                        = ((hour % 12) * 30 + (minute / 2) - munite * 6) % 360
                        = (30 * (hour % 12 )  - 5.5 * minute) % 360

      angle of hour = angle of hour + angle of minute passed
                    = (hour % 12) / 12 * 360 + (minute / 60) * (1/12 * 360)

      angle of minute = minute / 60 * 360


Test
------------------------

::

    $ python angle.py -v
    $ nosetests angle.py

QA
------------------------

::

    $ pep8 angle.py
    $ pylint angle.py
