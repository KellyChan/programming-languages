###################################
Nao
###################################

This is the repo for storing the demos of Robot Nao.

- Choregraphe
- Python

---------------------------
Choregraphe
---------------------------

How to use:

- import ``*.crg`` file into Choregraphe
- connect the robot
- play

---------------------------
Naoqi
---------------------------

How to use (remote):

- update the ``ip`` and ``port``
- type ``python xxx.py``

How to use (robot):

- upload the script on the robot, e.g. ``/home/nao/reacting_to_events.py``
- edit ``/home/nao/naoqipreferences/autoload.ini`` to have

::

    [python]
    /home/nao/reacting_to_events.py


---------------------------
References
---------------------------

- `Aldebaran Docs`_

.. _`Aldebaran Docs`: http://doc.aldebaran.com/2-1/index.html
