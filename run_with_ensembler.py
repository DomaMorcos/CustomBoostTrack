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
    
    # Arguments already defined in make_parser() (from args.py)
    existing_args = [action.dest for action in parser._actions]
    
    # Override defaults for existing arguments
    if 'iou_thresh' in existing_args:
        parser.set_defaults(iou_thresh=0.6)  # Override for WBF (was 0.3 in args.py)
    if 'min_hits' in existing_args:
        parser.set_defaults(min_hits=3)  # Same as default in args.py
    if 'conf' in existing_args:
        parser.set_defaults(conf=0.3)  # Map to conf_thresh
    
    # Add new arguments only if not already defined
    new_args = [
        ("--conf_thresh", {"type": float, "default": 0.3, "help": "Confidence threshold for fused boxes"}),
        ("--max_age", {"type": int, "default": None, "help": "Max age for tracks (overrides frame_rate if set)"}),
        ("--det_thresh", {"type": float, "default": None, "help": "Detection confidence threshold"}),
        ("--iou_threshold", {"type": float, "default": 0.35, "help": "IoU threshold for tracking"}),
        ("--min_box_area", {"type": float, "default": 10, "help": "Min box area for valid detections"}),
        ("--aspect_ratio_thresh", {"type": float, "default": 1.6, "help": "Aspect ratio threshold for detections"}),
        ("--lambda_iou", {"type": float, "default": 0.5, "help": "IoU cost weight"}),
        ("--lambda_mhd", {"type": float, "default": 0.25, "help": "Mahalanobis distance cost weight"}),
        ("--lambda_shape", {"type": float, "default": 0.25, "help": "Shape similarity cost weight"}),
        ("--use_dlo_boost", {"type": int, "default": 1, "help": "Enable DLO boost (1=True, 0=False)"}),
        ("--use_duo_boost", {"type": int, "default": 1, "help": "Enable DUO boost (1=True, 0=False)"}),
        ("--dlo_boost_coef", {"type": float, "default": None, "help": "DLO boost coefficient"}),
        ("--n_min", {"type": int, "default": 25, "help": "Min frames for interpolation"}),
        ("--n_dti", {"type": int, "default": 20, "help": "Max frame gap for linear interpolation"}),
        ("--interval", {"type": int, "default": 1000, "help": "Max frame gap for interpolation"}),
        ("--dataset", {"type": str, "default": "mot17"}),
        ("--result_folder", {"type": str, "default": "results/trackers/"}),
        ("--test_dataset", {"action": "store_true"}),
        ("--exp_name", {"type": str, "default": "test"}),
        ("--no_reid", {"action": "store_true"}),
        ("--no_cmc", {"action": "store_true"}),
        ("--s_sim_corr", {"action": "store_true"}),
        ("--btpp_arg_iou_boost", {"action": "store_true"}),
        ("--btpp_arg_no_sb", {"action": "store_true"}),
        ("--btpp_arg_no_vt", {"action": "store_true"}),
        ("--no_post", {"action": "store_true"}),
        ("--dataset_path", {"type": str}),
        ("--model1_path", {"type": str}),
        ("--model1_weight", {"type": float, "default": 0.5}),
        ("--model2_path", {"type": str}),
        ("--model2_weight", {"type": float, "default": 0.5}),
        ("--reid_path", {"type": str}),
        ("--reid_path2", {"type": str, "default": None}),
        ("--reid_weight1", {"type": float, "default": 0.5}),
        ("--reid_weight2", {"type": float, "default": 0.5}),
        ("--frame_rate", {"type": int, "default": 25}),
        ("--gnn_model_path", {"type": str, "default": None, "help": "Path to pretrained GNN model"}),
    ]
    
    for arg_name, kwargs in new_args:
        arg_dest = arg_name.lstrip('-').replace('-', '_')
        if arg_dest not in existing_args:
            parser.add_argument(arg_name, **kwargs)
    
    args = parser.parse_args()
    if args.dataset == "mot17":
        args.result_folder = os.path.join(args.result_folder, "MOT17-val")
    elif args.dataset == "mot20":
        args.result_folder = os.path.join(args.result_folder, "MOT20-val")
    elif args.dataset == "tarsh":
        args.result_folder = os.path.join(args.result_folder, "tarsh-val")
    
    if args.test_dataset:
        args.result_folder = args.result_folder.replace("-val", "-test")
    return args

def my_data_loader(main_path):
    img_pathes = [os.path.join(main_path, img) for img in os.listdir(main_path)]
    img_pathes = sorted(img_pathes)
    preproc = dataset.ValTransform(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    for idx, img_path in enumerate(img_pathes[:], 1):
        np_img = cv2.imread(img_path)
        # get size of image
        height, width, _ = np_img.shape
        img, target = preproc(np_img, None, (height, width))
        yield ((img.reshape(1, *img.shape), np_img), target, (height, width, torch.tensor(idx), None, ["test"]), None)

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
            tracker = BoostTrack(video_name=video_name, gnn_model_path=args.gnn_model_path)

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