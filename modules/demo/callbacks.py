import cv2
import numpy as np
from modules.filter.filter import predict

def imageCallback(image_rgb, vpr_module, sign_detection_module):
    landmark_match = vpr_module.match(image_rgb)
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    bboxes, labels, img_viz = sign_detection_module.detect_and_recognize(img_bgr)
    sign_detected = None
    if(labels is not None):
        for label in labels:
            if label in ["30", "40", "50", "60"]:
                sign_detected = int(label) # TODO: This does not make use of all of possible signs detected. Improve.
                break
    return landmark_match, sign_detected

def odometerCallback(hypotheses, odometer_df, noise_stdev=0.1):
    x_delta = np.sum(odometer_df["x"])
    offset = x_delta
    for idx, hypothesis in hypotheses.get().items():
        predict(hypothesis, offset, noise_stdev**2)
    return
