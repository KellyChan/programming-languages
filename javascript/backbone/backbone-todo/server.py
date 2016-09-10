import json

import web
from models import Todos

urls = (
    '/', 'index',
    '/todo', 'todo',
    '/todo/(\d*)', 'todo',
    '/todos/', 'todos',
)

app = web.application(urls, globals())

render = web.template.render('')


class index:
    def GET(self):
        return render.index()

class todo:
    def GET(self, todo_id=None):
        result = None
        itertodo = Todos.get_by_id(id=todo_id)
        for todo in itertodo:
            result = {
                "id": todo.id,
                "title": todo.title,
                "order": todo._order,
                "done": todo.done == 1,
            }
        return json.dumps(result)

    def POST(self):
        data = web.data()
        todo = json.loads(data)
        todo['_order'] = todo.pop('order')
        Todos.create(**todo)

    def PUT(self, todo_id=None):
        data = web.data()
        todo = json.loads(data)
        todo['_order'] = todo.pop('order')
        Todos.update(**todo)

    def DELETE(self, todo_id=None):
        Todos.delete(id=todo_id)


class todos:
    def GET(self):
        todos = []
        itertodos = Todos.get_all()
        for todo in itertodos:
            todos.append({
                "id": todo.id,
                "title": todo.title,
                "order": todo._order,
                "done": todo.done == 1,
            })
        return json.dumps(todos)

if __name__ == "__main__":
    app.run()