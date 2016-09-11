#----------------------------------------------------------------#
# Face++ API

# Create an app: http://www.faceplusplus.com.cn/create-a-new-app/
# Create the API_KEY and API_SECRET
API_KEY = 'YOUR API KEY'
API_SECRET = 'YOUR API SECRET'

#----------------------------------------------------------------#
# Images

# images for training
# image data types
# Leonardo DiCaprio
#  {'face': [{'attribute': {'age': {'range': 5, 'value': 24},
#                           'gender': {'confidence': 99.9998,
#                                      'value': 'Male'},
#                           'race': {'confidence': 99.5828,
#                                    'value': 'White'},
#                           'smiling': {'value': 6.56214}},
#             'face_id': 'd1374d61ddc5e04847e4b9a8ccba5132',
#             'position': {'center': {'x': 42.75, 'y': 49.916388},
#                          'eye_left': {'x': 32.551, 'y': 44.078428},
#                          'eye_right': {'x': 51.022667, 'y': 42.1801},
#                          'height': 38.628763,
#                          'mouth_left': {'x': 36.497333, 'y': 61.53495},
#                          'mouth_right': {'x': 50.798, 'y': 61.055017},
#                          'nose': {'x': 42.608167, 'y': 53.22592},
#                          'width': 38.5},
#             'tag': ''}],
#   'img_height': 598,
#   'img_id': '9103c3f0355d0a52065d7c3304abd925',
#   'img_width': 600,
#   'session_id': '881f8c7b002a424c9973835458265487',
#   'url': 'http://cn.faceplusplus.com/static/resources/python_demo/2.jpg'}

# training sets
IMAGE_DIR = 'http://cn.faceplusplus.com/static/resources/python_demo/'
PERSONS = [
    ('Jim Parsons', IMAGE_DIR + '1.jpg'),
    ('Leonardo DiCaprio', IMAGE_DIR + '2.jpg'),
    ('Andy Liu', IMAGE_DIR + '3.jpg')
]

# target images
TARGET_IMAGE = IMAGE_DIR + '4.jpg'
