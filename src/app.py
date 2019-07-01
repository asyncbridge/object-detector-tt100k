import sys
sys.path.insert(0,'../tt100k/fpgan/code/caffe/python')
import caffe
import os
import flask
from flask import request
from flask import Response
import time
import logging
import json
import datetime
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
from PIL import Image
from io import BytesIO
import exifutil
import base64
import random
import cv2
import skimage.io as io
import numpy as np
import anno_func
import pylab as pl

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "/tmp/tt100k_detection_uploads"
DETECTED_IMAGE_OUTPUT_WIDTH = 512
TT100K_IMAGE_SIZE = 2048

app = flask.Flask(__name__)
def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_output_width = DETECTED_IMAGE_OUTPUT_WIDTH

    if image.shape[1] < image_output_width:
       image_output_width = image.shape[1]

    image_pil = image
    #image_pil = resize_image(image, width=image_output_width)
    #image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = Image.fromarray((image_pil).astype('uint8'))
    #image_pil = image_pil.resize((256, 256))
    stream = BytesIO()
    image_pil.save(stream, format='png')
    #data = string_buf.getvalue().encode('base64').replace('\n', '')
    data = base64.b64encode(stream.getvalue()).decode('utf-8').replace('\n', '')
    return 'data:image/png;base64,' + data

def http_error_response(error_msg, status_code):
    data = {
        'status_code': status_code,
        'msg': error_msg
    }

    js = json.dumps(data)

    res = Response(js, status=status_code, mimetype='application/json')
    logger.error(error_msg)
    return res
	
def http_success_response(message, result):
    data = {
        'status_code': 200,
        'msg' : message,
        'result' : result
    }

    js = json.dumps(data)

    res = Response(js, status=200, mimetype='application/json')
    return res

class ObjectDetector(object):
    def __init__(self, model_path, weights_path, annotation_path, binaryproto_path, gpu_mode=True, gpu_id=0):
        logging.info('Loading Caffe and associated files.')

        self.model_path       = model_path 
        self.weights_path     = weights_path
        self.annotation_path  = annotation_path 
        self.binaryproto_path = binaryproto_path
        self.gpu_mode         = gpu_mode

        self.annos = json.loads(open(self.annotation_path).read())

        if self.gpu_mode:
           caffe.set_mode_gpu()
        else:
           caffe.set_mode_cpu()

        caffe.set_device(gpu_id)
        logging.info('Using GPU ID: ', gpu_id)
        self.net = caffe.Net(model_path, weights_path, caffe.TEST)

        logging.info('Loading test_mean.binaryproto file.')
        self.mean = caffe.proto.caffe_pb2.BlobProto.FromString(open(self.binaryproto_path).read())
        self.mn = np.array(self.mean.data)
        self.mn = self.mn.reshape(self.mean.channels, self.mean.height, self.mean.width)
        self.mn = self.mn.transpose((1,2,0))

    def detect(self, image_file_path):
        logging.info('Running detect.')
 
        imgdata = pl.imread(image_file_path)

        if imgdata.shape[1] != TT100K_IMAGE_SIZE or imgdata.shape[0] != TT100K_IMAGE_SIZE:
           imgdata = cv2.resize(imgdata, (TT100K_IMAGE_SIZE, TT100K_IMAGE_SIZE))
    
        if imgdata.max() > 2:
           imgdata = imgdata/255.        

        imgdata = imgdata[:,:,[2,1,0]]*255. - self.mn        

        rectmap = {}
        all_rect = []        

        logging.info('Running work.')
        anno_func.work(self.net, imgdata, all_rect)
        rectmap[0] = all_rect

        logging.info('Running get_refine_rects.')
        result_annos = anno_func.get_refine_rects(self.annos, rectmap)
        #detected_result = json.dumps(result_annos)
        detected_result = result_annos
        #print(detected_result)
        logging.info(detected_result)

        return detected_result
 
    '''
        darknet.set_gpu(0)
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.metadata_path = metadata_path
        self.net = darknet.load_net(cfg_path, weights_path, 0)
        self.meta = darknet.load_meta(metadata_path)
        
    def detect(self, image_file_path):
        detected_result = darknet.detect_objects(self.net, self.meta, self.cfg_path, self.weights_path, self.metadata_path, image_file_path)
        return detected_result
    '''
@app.route('/detect_object', methods=['POST'])
def detect_object():
    current_time = time.time()

    json_result = {
        "data": []
    }

    if flask.request.files['submitImageFile'].filename == '':
        return http_error_response("There is no image file.", 412)      

    try:
        imagefile = flask.request.files['submitImageFile']

        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                    werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)

        imagefile.save(filename)
        logging.info('Saving to %s.', filename)

        # Convert wrong orientation of uploaded image
        im_arr = exifutil.open_oriented_im(filename)
        
        image_file_path =  filename.encode('utf-8')
        logging.info('image_file_path is %s.', image_file_path)

        # Detect the objects on uploaded image
        detected_result = app.detector.detect(image_file_path)

        if request.args.get('embed_image') == "true":
           detected_image = showAnnsBBox(im_arr, detected_result)
           json_result["data"].append(dict(detected_embed_image=embed_image_html(detected_image)))


        json_result["data"].append(dict(detected_results=detected_result))

        if os.path.exists(filename):
            os.remove(filename)
            logging.info('Uploaded image file is removed: %s', filename)

    except Exception as err:
        logging.info('Upload the image error: %s', err)

        if os.path.exists(filename):
            os.remove(filename)
            logging.info('Uploaded image file is removed: %s', filename)

        return http_error_response(err, 400)

    elapsed_time = time.time() - current_time
    json_result["data"].append(dict(elapsed_time=elapsed_time))
		
    return http_success_response("success", json_result)

@app.route('/', methods=['GET'])
def index():
    json_result = {
        "deep_learning_framework":"Caffe, http://caffe.berkeleyvision.org",
        "model":"FPN-CNN model which is trained by tsinghua-tencent 100K dataset"
    }
    return http_success_response("success", json_result)

def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    logging.info("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()

def start_from_terminal(app):
    """
        Parse command line options and start the server.
        """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5003)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=True)

    opts, args = parser.parse_args()

    model_path = os.path.join('../tt100k/weights', 'model.prototxt').encode('utf-8')
    weights_path = os.path.join('../tt100k/weights', 'model.caffemodel').encode('utf-8')
    annotation_path = os.path.join('../tt100k/data', 'annotations.json').encode('utf-8') 
    binaryproto_path = os.path.join('../tt100k/data/lmdb', 'test_mean.binaryproto').encode('utf-8')   

    logging.info('model_path is %s.', model_path)
    logging.info('weights_path is %s.', weights_path)
    logging.info('annotation_path is %s.', annotation_path)
    logging.info('binaryproto_path is %s.', binaryproto_path)

     # Initialize tt100k object detector for warm start
    app.detector = ObjectDetector(model_path, weights_path, annotation_path, binaryproto_path, opts.gpu)    
        

    '''
    dirname = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(dirname, 'yolov3-cg.cfg').encode('utf-8')
    weights_path = os.path.join(dirname, 'yolov3-cg_final.weights').encode('utf-8')
    metadata_path = os.path.join(dirname, 'cg.data').encode('utf-8')

    logging.info('cfg_path is %s.', cfg_path)
    logging.info('weights_path is %s.', weights_path)
    logging.info('metadata_path is %s.', metadata_path)

    # Initialize tt100k object detector for warm start
    app.detector = ObjectDetector(cfg_path, weights_path, metadata_path)
    '''
    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    start_from_terminal(app)
