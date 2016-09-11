from clientbase import FaceClientBase

class Train(FaceClientBase):

    def __init__(self):
        super(Train, self).__init__()

    def identify(self, group_name):
        """
        Returns

            {'session_id': '8f7b3dc8d8e74d8a91bd94224637bf6b'}
        """
        return self.api.train.identify(group_name=group_name)

    def recognize(self):
        pass

    def verify(self):
        pass

    def search(self):
        pass

    def group_search(self):
        pass
