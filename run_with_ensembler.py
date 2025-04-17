import os
import shutil
import time

import dataset
import utils
from args import make_parser
from default_settings import GeneralSettings, get_detector_path_and_im_size, BoostTrackPlusPlusSettings, BoostTrackSettings
from external.adaptors import detector
from tracker.GBI import GBInterpolation
from tracker.boost_track import BoostTrack
from ultralytics import YOLO
import cv2
import numpy as np
import torch

from detectors import *

"""
Script modified from Deep OC-SORT: 
https://github.com/GerardMaggiolino/Deep-OC-SORT
"""


def get_main_args():
    parser = make_parser()
    parser.add_argument("--dataset", type=str, default="mot17")
    parser.add_argument("--result_folder", type=str, default="results/trackers/")
    parser.add_argument("--test_dataset", action="store_true")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--no_reid", action="store_true")
    parser.add_argument("--no_cmc", action="store_true")
    parser.add_argument("--s_sim_corr", action="store_true")
    parser.add_argument("--btpp_arg_iou_boost", action="store_true")
    parser.add_argument("--btpp_arg_no_sb", action="store_true")
    parser.add_argument("--btpp_arg_no_vt", action="store_true")
    parser.add_argument("--no_post", action="store_true")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model1_path", type=str)
    parser.add_argument("--model1_weight", type=float, default=0.5)
    parser.add_argument("--model2_path", type=str)
    parser.add_argument("--model2_weight", type=float, default=0.5)
    parser.add_argument("--reid_path", type=str)
    parser.add_argument("--reid_path2", type=str, default=None)
    parser.add_argument("--reid_weight1", type=float, default=0.5)
    parser.add_argument("--reid_weight2", type=float, default=0.5)
    parser.add_argument("--frame_rate", type=int, default=25)
    
    # Detector parameters
    parser.add_argument("--iou_thresh", type=float, default=0.6, help="IoU threshold for WBF")
    parser.add_argument("--conf_thresh", type=float, default=0.3, help="Confidence threshold for fused boxes")
    
    # Tracker parameters
    parser.add_argument("--max_age", type=int, default=None, help="Max age for tracks (overrides frame_rate if set)")
    parser.add_argument("--min_hits", type=int, default=3, help="Min hits to confirm a track")
    parser.add_argument("--det_thresh", type=float, default=None, help="Detection confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.35, help="IoU threshold for tracking")
    parser.add_argument("--min_box_area", type=float, default=10, help="Min box area for valid detections")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="Aspect ratio threshold for detections")
    parser.add_argument("--lambda_iou", type=float, default=0.5, help="IoU cost weight")
    parser.add_argument("--lambda_mhd", type=float, default=0.25, help="Mahalanobis distance cost weight")
    parser.add_argument("--lambda_shape", type=float, default=0.25, help="Shape similarity cost weight")
    parser.add_argument("--use_dlo_boost", type=int, default=1, help="Enable DLO boost (1=True, 0=False)")
    parser.add_argument("--use_duo_boost", type=int, default=1, help="Enable DUO boost (1=True, 0=False)")
    parser.add_argument("--dlo_boost_coef", type=float, default=None, help="DLO boost coefficient")
    
    # Post-processing parameters
    parser.add_argument("--n_min", type=int, default=25, help="Min frames for interpolation")
    parser.add_argument("--n_dti", type=int, default=20, help="Max frame gap for linear interpolation")
    parser.add_argument("--interval", type=int, default=1000, help="Max frame gap for interpolation")
    
    args = parser.parse_args()
    if args.dataset == "mot17":
        args.result_folder = os.path.join(args.result_folder, "MOT17-val")
    elif args.dataset == "mot20":
        args.result_folder = os.path.join(args.result_folder, "MOT20-val")
    elif args.dataset == "tarsh":  # Add custom dataset
        args.result_folder = os.path.join(args.result_folder, "tarsh-val")
    
    if args.test_dataset:
        args.result_folder = args.result_folder.replace("-val", "-test")
    return args

def main():
    args = get_main_args()
    GeneralSettings.values['dataset'] = args.dataset
    GeneralSettings.values['use_embedding'] = not args.no_reid
    GeneralSettings.values['use_ecc'] = not args.no_cmc
    GeneralSettings.values['test_dataset'] = args.test_dataset
    GeneralSettings.values['reid_path'] = args.reid_path
    GeneralSettings.values['min_hits'] = args.min_hits
    GeneralSettings.values['iou_threshold'] = args.iou_threshold
    GeneralSettings.values['min_box_area'] = args.min_box_area
    GeneralSettings.values['aspect_ratio_thresh'] = args.aspect_ratio_thresh
    if args.max_age is not None:
        GeneralSettings.values['max_age'] = args.max_age
    else:
        GeneralSettings.values['max_age'] = args.frame_rate
    
    if args.det_thresh is not None:
        GeneralSettings.dataset_specific_settings[args.dataset] = {'det_thresh': args.det_thresh}
    if args.dlo_boost_coef is not None:
        BoostTrackSettings.dataset_specific_settings[args.dataset] = {'dlo_boost_coef': args.dlo_boost_coef}
    
    BoostTrackSettings.values['s_sim_corr'] = args.s_sim_corr
    BoostTrackSettings.values['lambda_iou'] = args.lambda_iou
    BoostTrackSettings.values['lambda_mhd'] = args.lambda_mhd
    BoostTrackSettings.values['lambda_shape'] = args.lambda_shape
    BoostTrackSettings.values['use_dlo_boost'] = bool(args.use_dlo_boost)
    BoostTrackSettings.values['use_duo_boost'] = bool(args.use_duo_boost)
    
    BoostTrackPlusPlusSettings.values['use_rich_s'] = not args.btpp_arg_iou_boost
    BoostTrackPlusPlusSettings.values['use_sb'] = not args.btpp_arg_no_sb
    BoostTrackPlusPlusSettings.values['use_vt'] = not args.btpp_arg_no_vt

    tracker = None
    results = {}
    frame_count = 0
    total_time = 0

    model1 = YoloDetector(args.model1_path)
    model2 = YoloDetector(args.model2_path)
    det = EnsembleDetector(model1, model2, args.model1_weight, args.model2_weight, args.iou_thresh, args.conf_thresh)
    
    for (img, np_img), _, info, _ in my_data_loader(args.dataset_path):
        frame_id = info[2].item()
        video_name = info[4][0].split("/")[0]
        tag = f"{video_name}:{frame_id}"
        if video_name not in results:
            results[video_name] = []

        print(f"Processing {video_name}:{frame_id}\r", end="")
        if frame_id == 1:
            print(f"Initializing tracker for {video_name}")
            print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
            if tracker is not None:
                tracker.dump_cache()
            tracker = BoostTrack(video_name=video_name)

        pred = det(np_img)
        start_time = time.time()
        if pred is None:
            continue
        targets = tracker.update(pred, img, np_img, tag)
        tlwhs, ids, confs = utils.filter_targets(targets, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])
        print(f"{len(ids)} ids detected")
        total_time += time.time() - start_time
        frame_count += 1
        results[video_name].append((frame_id, tlwhs, ids, confs))

    print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
    tracker.dump_cache()
    folder = os.path.join(args.result_folder, args.exp_name, "data")
    os.makedirs(folder, exist_ok=True)
    for name, res in results.items():
        result_filename = os.path.join(folder, f"{name}.txt")
        utils.write_results_no_score(result_filename, res)
    print(f"Finished, results saved to {folder}")

    if not args.no_post:
        post_folder = os.path.join(args.result_folder, args.exp_name + "_post")
        pre_folder = os.path.join(args.result_folder, args.exp_name)
        if os.path.exists(post_folder):
            print(f"Overwriting previous results in {post_folder}")
            shutil.rmtree(post_folder)
        shutil.copytree(pre_folder, post_folder)
        post_folder_data = os.path.join(post_folder, "data")
        utils.dti(post_folder_data, post_folder_data, n_dti=args.n_dti, n_min=args.n_min)

        print(f"Linear interpolation post-processing applied, saved to {post_folder_data}.")

        post_folder_gbi = os.path.join(args.result_folder, args.exp_name + "_post_gbi", "data")
        if not os.path.exists(post_folder_gbi):
            os.makedirs(post_folder_gbi)
        for file_name in os.listdir(post_folder_data):
            in_path = os.path.join(post_folder_data, file_name)
            out_path2 = os.path.join(post_folder_gbi, file_name)
            GBInterpolation(path_in=in_path, path_out=out_path2, interval=args.interval)
        print(f"Gradient boosting interpolation post-processing applied, saved to {post_folder_gbi}.")

if __name__ == "__main__":
    main()