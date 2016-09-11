from faceclient.client import FaceClient


class Demo(object):

    def __init__(self):
        self.client = FaceClient()

    def create_images(self, group_name):
        faces = self.client.detection.detect()
        for name, face in faces.iteritems():
            self.client.person.create(person_name=name,
                                      face_id=face['face'][0]['face_id'])

        self.client.group.create(group_name=group_name)
        self.client.group.add_person(group_name=group_name,
                                     person_name=faces.iterkeys())
        return faces

    def train(self, group_name):
        return self.client.train.identify(group_name=group_name)

    def recognize(self, group_name):
        return self.client.recognition.identify(group_name=group_name)

    def delete_group(self, group_name):
        self.client.group.delete(group_name=group_name)

    def delete_person(self, person_names):
        self.client.person.delete(person_name=person_names)

    def callback(self, group_name, person_names):
        self.client.group.delete(group_name=group_name)
        self.client.person.delete(person_name=person_names)
