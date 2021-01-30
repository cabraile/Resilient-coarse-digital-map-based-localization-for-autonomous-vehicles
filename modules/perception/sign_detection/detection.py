import cv2
import time
import os
import numpy as np

from modules.perception.sign_detection.utils import *
from modules.perception.sign_detection.utils.utils import *
from modules.perception.sign_detection.utils.torch_utils import *

from modules.perception.sign_detection.models import *

import torch

from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog

import os


class BoundingBox:

	def __init__(self, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, prob, class_name):
		self.p1 = (p1x, p1y)
		self.p2 = (p2x, p2y)
		self.p3 = (p3x, p3y)
		self.p4 = (p4x, p4y)
		self.prob = prob
		self.class_name = class_name
		return


class SignDetectionModule:

	def __init__(self, cfg_file_path, weights_path, threshold=0.89):

		"""
		@param cfg_file_path: str
		@param weights_path: str
		"""
		self.config_recognition = {
			"image_size" : 128,
			"nclasses" : 5,
			"root_ds_dir" : "/absolute/path/to/repository/modules/perception/street_signs_dataset"
		}
		
		self.config_detection = {
			"image_size" : 416,
			"nms_thres" : 0.5, # iou threshold for non-maximum suppression'
			"threshold" : threshold
		} 

		self.device = select_device(force_cpu=True, apex=False)
		torch.backends.cudnn.benchmark = False  # set False for reproducible results
		self.detection_model = Darknet(cfg_file_path, self.config_detection["image_size"])
		if weights_path.endswith('.pt'):  # pytorch format
			self.detection_model.load_state_dict(torch.load(weights_path, map_location=self.device)['model'])
		else:
			_ = load_darknet_weights(self.detection_model, weights_path)

		# Eval mode
		#print("HERE HERE HERE HERE")
		self.detection_model.to(self.device).eval()
		self.class_to_ids = {"30":0, "40":1, "50":2, "60":3, "not_park":4}
		self.ids_to_class = {0:"30", 1:"40", 2:"50", 3:"60", 4:"not_park"}
		self.colors = {
			'30':(0, 0, 255), '40': (0,255,0),
			'50':(255,255,0), '60': (122,0,0), 
			'not_park':(255,122,100)
		}

		self.load_recognition_model()
		return

	def preprocess_recognition(self, img_bgr, equalize):
		height = self.config_recognition["image_size"]
		width = self.config_recognition["image_size"]
		if(equalize):
			img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
			img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
			img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
		image_from_array = Image.fromarray(img_bgr, 'RGB')
		size_image = image_from_array.resize((height, width))
		image = np.array(size_image) # BGR!
		return image
	
	def load_recognition_model(self):
		
		data=None
		labels=[]

		height = self.config_recognition["image_size"]
		width = self.config_recognition["image_size"]
		classes = self.config_recognition["nclasses"]

		root_path = self.config_recognition["root_ds_dir"]
		Class_names =os.listdir(root_path)
		class_ids = {"30":0, "40":1, "50":2, "60":3, "not_park":4}
		for name in Class_names:
			path = root_path + "/" + name
			Class=os.listdir(path)
			for a in Class:
				try:
					ext = a.split(".")[-1]
					if ext != "ppm" and ext != "jpg" and ext != "png" and ext != "jpeg":
						continue
					image_bgr = cv2.imread(path+"/"+a)
					image_rgb = self.preprocess_recognition(image_bgr,equalize=False)[:,:,::-1]
					descriptor = hog(image_rgb, orientations=8, pixels_per_cell=(8,8),
							cells_per_block=(1, 1), visualize=False, feature_vector=True, multichannel=True)
					descriptor = descriptor / np.linalg.norm(descriptor)
					if(data is None):
						data = descriptor
					else:
						data = np.vstack((data,descriptor))
					labels.append(class_ids[name])
				except AttributeError:
					print(" ")

		Cells=np.array(data)
		labels=np.array(labels)

		#Randomize the order of the input images
		s=np.arange(Cells.shape[0])
		np.random.seed(classes)
		np.random.shuffle(s)
		Cells=Cells[s]
		labels=labels[s]

		X_train=Cells[:(int)(1.0*len(labels))]
		y_train=labels[:(int)(1.0*len(labels))]

		self.recognition_model = KNeighborsClassifier(n_neighbors=5)
		self.recognition_model.fit(X_train, y_train)
		return

	def letterbox(self, img_l, new_shape=416, color=(128, 128, 128), mode='auto'):
		# Resize a rectangular image to a 32 pixel multiple rectangle
		# https://github.com/ultralytics/yolov3/issues/232
		shape = img_l.shape[:2]  # current shape [height, width]

		if isinstance(new_shape, int):
			ratio = float(new_shape) / max(shape)
		else:
			ratio = max(new_shape) / max(shape)  # ratio  = new / old
		ratiow, ratioh = ratio, ratio
		new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

		# Compute padding https://github.com/ultralytics/yolov3/issues/232
		if mode == 'auto':  # minimum rectangle
			dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
			dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
		elif mode == 'square':  # square
			dw = (new_shape - new_unpad[0]) / 2  # width padding
			dh = (new_shape - new_unpad[1]) / 2  # height padding
		elif mode == 'rect':  # square
			dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
			dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
		elif mode == 'scaleFill':
			dw, dh = 0.0, 0.0
			new_unpad = (new_shape, new_shape)
			ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

		if shape[::-1] != new_unpad:  # resize
			img_l = cv2.resize(img_l, new_unpad, interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
		top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
		left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
		img_l = cv2.copyMakeBorder(img_l, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
		return img_l, ratiow, ratioh, dw, dh

	def detect_and_recognize(self, img_bgr, conf_thres=None, nms_thres=0.5):
		if(conf_thres is None):
			conf_thres = self.config_detection["threshold"]
		img, _,_,_,_ = self.letterbox(img_bgr, new_shape=self.config_detection["image_size"])
		
		# Normalize RGB
		img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
		img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		
		# Get detections
		img = torch.from_numpy(img).unsqueeze(0).to(self.device)

		pred, _ = self.detection_model(img)
		det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]
		bboxes = None
		img_viz = None
		labels = None
		if det is not None and len(det) > 0:
			det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_bgr.shape).round()
			bboxes = self.process_detect(det, img_bgr)
			labels, img_viz = self.recognition(img_bgr,bboxes)
		return bboxes, labels, img_viz
				
	def recognition(self, img_bgr, bboxes):
		labels = []
		img_viz = np.copy(img_bgr)
		for bbox in bboxes:
			p1 = bbox.p1; p3 = bbox.p3
			img_sign_bgr = img_bgr[p1[1]:p3[1], p1[0]:p3[0],:]
			img_sign_rgb = self.preprocess_recognition(img_sign_bgr,equalize=True)[:,:,::-1]
			descriptor = hog(img_sign_rgb, orientations=8, pixels_per_cell=(8,8),
				cells_per_block=(1, 1), visualize=False, feature_vector=True, multichannel=True)
			descriptor = descriptor / np.linalg.norm(descriptor)
			res = self.recognition_model.predict([descriptor])
			label = self.ids_to_class[res[0]]
			labels.append(label)

			img_viz = cv2.rectangle(img_viz, (p1[0], p1[1]), (p3[0],p3[1]), self.colors[label], 2)
			img_viz = cv2.putText(img_viz, label, (p1[0] - 2, p1[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors[label], 1, cv2.LINE_AA)
		return labels, img_viz

	def process_detect(self, objects, img):

		bounding_array = []  
		for x0, x1, x2, x3, conf, cls_conf, _cls  in objects:

			label = self.ids_to_class[int(_cls)]

			prob_c = conf
			
			p1x= int(x0)
			p1y= int(x1)
		
			p2x= int(x0)
			p2y= int(x3)
		
			p3x= int(x2)
			p3y= int(x3)
		
			p4x= int(x2)
			p4y= int(x1)

			b = BoundingBox(p1x, p1y,p2x, p2y, p3x, p3y, p4x, p4y, float(prob_c), int(_cls))
			bounding_array.append(b)

		return bounding_array



if __name__=='__main__':
	import glob
	import PIL

	root_dir = "yolo/v3/dir"
	test_dir = "path/to/images/test"
	
	list_img = glob.glob(test_dir + "/*.png")
	detector = SignDetectionModule(
		cfg_file_path=root_dir+"/yolov3-tiny-obj.cfg",
		weights_path=root_dir+"/yolov3-tiny-obj_5000.weights",
	)
	
	for test_image_path in list_img:
		img = cv2.imread(test_image_path)
		print("\r"+test_image_path, end="")

		bboxes, labels, img_viz = detector.detect_and_recognize(img,0.89,0.5)
		if(img_viz is not None):
			cv2.imshow("Detections", img_viz)
			cv2.waitKey(0)
	exit(0)
	