import settings.settings as settings
from clientbase import FaceClientBase

class Recognition(FaceClientBase):

    def __init__(self):
        super(Recognition, self).__init__()

    def train(self):
        pass

    def verify(self):
        pass

    def identify(self, group_name, target_url=settings.TARGET_IMAGE):
        """
        Returns

            {'face': [{'candidate': [{'confidence': 12.613228,
                                      'person_id': '2e468cb0ab6eaa35e5f5a36f2ee7553e',
                                      'person_name': 'Jim Parsons',
                                      'tag': ''},
                                     {'confidence': 2.212482,
                                      'person_id': '908ca715eb619c5317ad076af70fe29e',
                                      'person_name': 'Leonardo DiCaprio',
                                      'tag': ''},
                                     {'confidence': 0.0,
                                      'person_id': '8dcfde75f523ab44e3c3f45ec022f6e5',
                                      'person_name': 'Andy Liu',
                                      'tag': ''}],
                       'face_id': 'b2008aa177b2c476572ca65d0084f13f',
                       'position': {'center': {'x': 54.522613, 'y': 26.333333},
                                    'eye_left': {'x': 47.511307, 'y': 23.069},
                                    'eye_right': {'x': 60.246985, 'y': 22.802},
                                    'height': 19.0,
                                    'mouth_left': {'x': 48.529648, 'y': 32.322333},
                                    'mouth_right': {'x': 59.838191,
                                                    'y': 32.016167},
                                    'nose': {'x': 55.24196, 'y': 27.1255},
                                    'width': 28.643216}}],
            'session_id': '9a58656da3b347bab0b3b56657dabbe4'}
        """

        return self.api.recognition.identify(group_name=group_name, url=target_url)

    def recognize(self):
        pass

    def compare(self):
        pass


    def search(self):
        pass

    def group_search(self):
        pass
