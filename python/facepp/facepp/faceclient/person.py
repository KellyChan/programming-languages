from clientbase import FaceClientBase

class Person(FaceClientBase):

    def __init__(self):
        super(Person, self).__init__()

    def create(self, person_name, face_id):
        return self.api.person.create(person_name=person_name, face_id=face_id)

    def delete(self, person_name):
        return self.api.person.delete(person_name=person_name)

    def add_face(self):
        pass

    def remove_face(self):
        pass

    def get_info(self):
        pass

    def set_info(self):
        pass
