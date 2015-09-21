Todo App
====

### Toolkit

- Python 2.7
- Django 1.7
- Twitter Bootstrap
- jQuery

### Functions

- add a new task
- remove a task
- update the status (undo/done)
- update the info (task, priority)


### Models

- user
- pubtime
- todo
- flag
    - 1 - undo
    - 0 - done
- priority
    - 1 - most important
    - 2 - important
    - 3 - not important


### Views


| View       | Model          | Templates            | Function                         |
|:-----------|:---------------|:---------------------|:---------------------------------|
| TodoView   |                | todo.html            | main/index                       |
| AddView    | todo, priority | todo.html, add.html  | add a new task                   |
| DeleteView |                | todo.html            | delete a task                    |
| DoneView   | flag           | todo.html            | update the status: undo -> done  |
| UndoView   | flag           | todo.html            | udpate the status: done -> undo  |
| EditView   | todo, priority | todo.html, edit.html | update todo, priority            |


### Files

    todosite/
        __init__.py
        wsgi.py
        settings.py
        urls.py
    todo/
        __init__.py
        migrations/
            __init__.py

        admin.py
        urls.py
        models.py
        views.py
        tests.py
        templates/
            todo/
            base.html
        static/
            todo/
                fonts/
                css/
                js/  

