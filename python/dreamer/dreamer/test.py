import matplotlib 
matplotlib.use('Agg')
import sys
sys.path.insert(0, '/usr/local/caffe/python/')
import caffe

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd

import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format


# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

