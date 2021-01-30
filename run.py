from modules.map.draw                               import draw_routes, draw_hypotheses, zoom_to
from modules.map.route                              import Route
from modules.filter.hypotheses                      import Hypotheses
from modules.filter.filter                          import predict, update, g_pdf
from modules.perception.holistic_feature_matcher    import HolisticFeatureMatcher
from modules.perception.sign_detection              import detection 

from modules.demo.data_manager                      import DataManager
from modules.demo.benchmark                         import print_statistics
from modules.demo.pdf                               import p_landmark_given_position_and_route, p_speedlimit_given_position_and_route
from modules.demo.callbacks                         import imageCallback, odometerCallback
from modules.demo.visualization                     import save_figure_map, save_figure_landmark, draw_figure_routes, draw_landmarks

import cv2  # Income image processing
import sys  # Args
import time # Benchmarking
import utm  # Conversion from latitude and longitude to UTM 
import yaml # Load all script parameters
import os   # Current dir for loading the sign detection module

import numpy as np

import matplotlib.pyplot as plt

def do_ignore(timestamp, params):
    if("dataset" in params):
        if("ignore_before_timestamp" in params["dataset"]):
            if(timestamp < params["dataset"]["ignore_before_timestamp"]):
                return True
    return False

def main(argv):
    
    if(len(argv) < 2):
        print("Error: arguments missing. Usage: python run.py /path/to/config.yaml")
        return -1

    print("[Main Scope] Loading parameters...")

    # 1. SETUP

    # 1.1 Load parameters
    params = None
    with open(sys.argv[1]) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    print("[Main Scope] Loaded parameters. Loading routes and branches...")

    # 1.2 Load all routes and their branches
    possible_routes = {}
    branches = {}
    for key, value in params["map"].items():
        if("route" in key):
            route_idx = int(''.join(filter(str.isdigit, key)))
            route = Route()
            route.from_yaml(value)
            route.idx = route_idx
            possible_routes[route_idx] = route
        elif("branch" in key):
            branch_idx = int(''.join(filter(str.isdigit, key)))
            route_idx = value[2]
            x, y, _, _ = utm.from_latlon(value[0], value[1])
            branches[branch_idx] = np.array([x,y,route_idx])

    # 1.3 Define the initial planned route
    active_routes = {
        0 : possible_routes[0]
    }
    inactive_routes = { }

    print("[Main Scope] Loaded routes and branches. Initializing hypothesis...")

    # 1.4 Initial hypothesis
    # Single variable gaussian distribution along the route length
    hypotheses = Hypotheses()
    hypotheses.create_hypothesis(
        mean = params["filter"]["initial_position"],
        variance = params["filter"]["initial_variance"],
        route = active_routes[0],
        weight=1.0
    )

    print("[Main Scope] Initialized hypothesis. Loading sensor data...")

    # 1.5 Load sensor data
    data_manager = DataManager(params["sensor"]["msgs"])

    print("[Main Scope] Done loading sensor data. Now started computing landmarks' feature vectors.")

    # 1.6 Compute landmark feature vectors for place recognition
    list_landmarks = []
    for route in possible_routes.values():
        list_landmarks = list_landmarks + route.__landmarks__["object"]
    fex_landmarks = HolisticFeatureMatcher(list_landmarks, model_name=params["sensor"]["settings"]["model"], threshold=params["sensor"]["settings"]["landmark_matching_threshold"])
    print("[Main Scope] Done computing feature vectors. Started main loop.")

    # 1.7 Loading sign detection and recognition model
    print("[Main Scope] Done computing feature vectors. Started loading the sign detection and recognition module.")

    root_sign_detection_dir = os.getcwd() + "/modules/perception/sign_detection/model"
    sign_detector = detection.SignDetectionModule(
		cfg_file_path=root_sign_detection_dir+"/yolov3-tiny-obj.cfg",
		weights_path=root_sign_detection_dir+"/yolov3-tiny-obj_5000.weights",
        threshold=params["sensor"]["settings"]["sign_recognition_threshold"]
	)

    print("[Main Scope] Done loading the sign detection and recognition module. Started main loop.")
    # 1.8 Control variables
    list_error = { "timestamp": [], "error": []}

    plt_fig = plt.figure(1,figsize=(15,10))
    plt_axes = {
        "global_scope"  : plt_fig.add_subplot(111),
    }

    step = 0

    # > used for computing the time statistics
    list_timestamp          = [] 
    list_benchmark_duration = { "prediction" : [], "update" : [], "detection": [] }
    time_localized          = 0
    last_localized_ts       = None
    last_groundtruth_xy     = None

    # > visualization
    last_image          = None
    detected_landmarks  = []
    
    while(data_manager.hasNext()):
        do_plot = False
        # 2 PROCESS INCOMING DATA

        # 2.1 Retrieve the most recent data
        data = data_manager.next()
        timestamp = data["timestamp"]
        if(do_ignore(timestamp, params)):
            print("\r[Main Scope] Ignored data from timestamp: {}".format(timestamp),end="")
            continue
        
        print("\r[Main Scope] Step {:04d}. Current timestamp: {}.".format(step,timestamp), end="")
        if(len(list_error["error"] ) > 4):
            _n = min(len(list_error["error"]),100)
            avg_error = np.average(list_error["error"][:-_n])
            print(" Average absolute error (from the last 100 estimations): {:.2f}m.".format(avg_error), end="")
        
        # 2.2 If image is available
        flag_detected_sign = False
        flag_matched_place = False
        landmark_match = None
        sign_recognized = None
        if(data["image"] is not None):
            image = data["image"]
            _time_start = time.time()
            landmark_match, sign_recognized = imageCallback(
                image, vpr_module=fex_landmarks, sign_detection_module=sign_detector
            )
            _duration = time.time() - _time_start
            list_benchmark_duration["detection"].append(_duration)
            # Detected landmark
            if(landmark_match is not None):
                flag_matched_place = True
                detected_landmarks.append(landmark_match)
            # Detected sign
            if(sign_recognized is not None):
                flag_detected_sign = True
                print("\n[ImageCallback] {}kmph sign detected!\n".format(sign_recognized))
            last_image = image

        # 2.3 Predict - if odometer available
        if(data["odometer"] is not None):
            _time_start = time.time()
            odometerCallback(hypotheses, data["odometer"])
            _duration = time.time() - _time_start
            list_benchmark_duration["prediction"].append(_duration)

        # 2.4 Update - if detection available
        flag_do_update = flag_detected_sign or flag_matched_place

        _time_start = time.time()
        if(flag_do_update):
            if(flag_detected_sign and flag_matched_place): # TODO: proper PDF when both are detected
                print("\n[Update] Detected sign and matched landmark")
                pass
            elif(flag_matched_place):
                print("\n[Update] Matched landmark")
                def p_y_given_x_r(x, r):
                    return p_landmark_given_position_and_route(landmark_match, x, r, params["sensor"]["settings"]["landmark_matching_sensitivity"], params["sensor"]["settings"]["landmark_matching_fpr"])
            else:
                print("\n[Update] Detected speed limit sign")
                def p_y_given_x_r(x,r):
                    return p_speedlimit_given_position_and_route(sign_recognized, x, r, params["sensor"]["settings"]["sign_recognition_sensitivity"], params["sensor"]["settings"]["sign_recognition_fpr"])

            # Update hypothesis
            for hypothesis in hypotheses.get().values():
                update(hypothesis, p_y_given_x_r, nsamples=100000)
                
            # Normalize hypothesis weights (range within [0,1])
            hypotheses.normalize_weights()
            
        # 3 PRUNE HYPOTHESES
        flag_pruned_branch = False
        removed_keys = hypotheses.prune_hypotheses(threshold=params["filter"]["prune_threshold"])
        for idx in removed_keys:
            print("[Main Scope] Pruned route {}".format(idx))
            flag_pruned_branch = True
            inactive_routes[idx] = active_routes[idx]
            del active_routes[idx]

        _duration = time.time() - _time_start
        list_benchmark_duration["update"].append(_duration)

        ## 4 BRANCH
        ## 4.1 Check if any of the hypotheses' mean is close to any branching position
        flag_expanded_branches = False
        expand_branches_ids = []
        closest_hypothesis = {} # Maps the branch to be expanded to the closest hypothesis
        for idx, hypothesis in hypotheses.__dict__.items():
            h_xy = np.array(hypothesis.route.from_distance_to_xy(hypothesis.mean))
            for branch_idx, branch in branches.items():
                branch_xy = np.array([branch[0], branch[1]])
                d = np.linalg.norm(h_xy - branch_xy)
                if (d < 20):
                    expand_branches_ids.append(branch_idx)
                    closest_hypothesis[branch_idx] = hypothesis
                    flag_expanded_branches = True

        # 4.2 Expand branches
        for idx in expand_branches_ids:
            route_idx = branches[idx][2] # The third element of the list is the route's idx the branch would expand
            active_routes[route_idx] = possible_routes[route_idx]
            hypothesis = closest_hypothesis[idx]
            hypotheses.create_hypothesis(
                mean        = hypothesis.mean,
                variance    = hypothesis.variance,
                route       = active_routes[route_idx], # Branch route to follow
                weight      = hypothesis.weight
            )
            del branches[idx]

        hypotheses.normalize_weights()
        if(len(active_routes) == 1):
            if(len(list_timestamp) != 0):
                time_localized += (timestamp - list_timestamp[-1]) * 1e-9
            last_localized_ts = timestamp
        
        # 5 VISUALIZATION
        
        # 5.1 Compute the absolute error between the estimation and the groundtruth
        if(data["groundtruth"] is not None):
            gt_coords = np.array(data["groundtruth"].to_numpy())
            if(len(gt_coords.shape) > 1 ):
                gt_coords = gt_coords[0,:]
            ret = utm.from_latlon(gt_coords[0], gt_coords[1])
            x = ret[0]
            y = ret[1]
            last_groundtruth_xy = np.array((x,y))

        if(last_groundtruth_xy is None):
            step += 1
            continue

        list_timestamp.append(data["timestamp"])
        best_hypothesis = hypotheses.get_best_hypothesis()
        curr_est_xy = np.array(best_hypothesis.route.from_distance_to_xy(best_hypothesis.mean))
        groundtruth_xy = last_groundtruth_xy

        abs_error = np.linalg.norm( np.array(groundtruth_xy) - np.array(curr_est_xy) )
        if(len(active_routes) == 1):
            list_error["timestamp"].append(timestamp)
            list_error["error"].append(abs_error)

        # 5.2 Plot global
        flag_display_iter = (step % params["visualization"]["skip_n_iters"] == 0)
        do_plot = flag_do_update or flag_expanded_branches or flag_pruned_branch
        if(do_plot or flag_display_iter):
            plt_fig.clf()
            plt_axes = {
                "global_scope"  : plt_fig.add_subplot(111),
            }

            # Route related axes            
            draw_figure_routes(plt_axes, hypotheses, groundtruth_xy, active_routes, inactive_routes, last_image[::4,::4,:], detected_landmarks, zoom=params["visualization"]["zoom_radius"])

            plt.pause(0.01)

        step += 1
    print("[Main Scope] Finished main loop.")
    plt.show()
    print_statistics(list_error, list_benchmark_duration, list_timestamp, time_localized)
    return 0


if __name__ == "__main__":
    exit(main(sys.argv))