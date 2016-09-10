Tetris
======

### 1. File Trees


folder structure

    .
    |---- run_game.py
    |---- /lib/
    |       |---- game.py
    |       |---- menu.py
    |       |---- main.py
    |       |---- tetris.py
    |       |---- shape.py
    |       |
    |       |---- sound.py
    |       |---- util.py
    |
    |---- /data/
            |---- /fonts/
            |---- /images/
            |---- /sounds/


function structure

    .
    |---- run_game.py
    |---- /lib/
    |       |---- game.py --- (sound/util)
    |       |       |---- menu.py --- (sound/util)
    |       |       |---- main.py
    |       |               |---- tetris.py
    |       |                        |---- shape.py --- (util.py)
    |       |
    |       |
    |       |---- sound.py
    |       |---- util.py



