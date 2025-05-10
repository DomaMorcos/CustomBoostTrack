import os
import shutil
import time
import math

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
        ("--visualize", {"action": "store_true", "help": "Enable visualization of detections and tracks"}),
        ("--track_percent", {"type": float, "default": 1.0, "help": "Percentage of video to track (0.0 to 1.0)"}),
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
    
    # Validate track_percent
    if not 0.0 < args.track_percent <= 1.0:
        raise ValueError("track_percent must be between 0.0 and 1.0")
    
    return args

def my_data_loader(main_path, track_percent=1.0):
    img_pathes = [os.path.join(main_path, img) for img in os.listdir(main_path)]
    img_pathes = sorted(img_pathes)
    # Limit to track_percent of images
    num_images = math.ceil(len(img_pathes) * track_percent)
    img_pathes = img_pathes[:num_images]
    preproc = dataset.ValTransform(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    for idx, img_path in enumerate(img_pathes[:], 1):
        np_img = cv2.imread(img_path)
        # get size of image
        height, width, _ = np_img.shape
        img, target = preproc(np_img, None, (height, width))
        yield ((img.reshape(1, *img.shape), np_img), target, (height, width, torch.tensor(idx), None, ["test"]), img_path)

def visualize_detections(np_img, model1_preds, model2_preds, ensemble_preds, video_name, frame_id, vis_folder):
    """Visualize detections from both YOLO models and ensemble, saving to disk."""
    img_model1 = np_img.copy()
    img_model2 = np_img.copy()
    img_ensemble = np_img.copy()
    
    # Draw model1 detections (YOLO) in blue
    if len(model1_preds) > 0:
        for pred in model1_preds:
            x1, y1, x2, y2, conf = pred.tolist()
            cv2.rectangle(img_model1, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img_model1, f"{conf:.2f}", (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw model2 detections (YOLO) in green
    if len(model2_preds) > 0:
        for pred in model2_preds:
            x1, y1, x2, y2, conf = pred.tolist()
            cv2.rectangle(img_model2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_model2, f"{conf:.2f}", (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw ensemble detections in red
    if len(ensemble_preds) > 0:
        for pred in ensemble_preds:
            x1, y1, x2, y2, conf = pred.tolist()
            cv2.rectangle(img_ensemble, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img_ensemble, f"{conf:.2f}", (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save images
    os.makedirs(os.path.join(vis_folder, video_name), exist_ok=True)
    cv2.imwrite(os.path.join(vis_folder, video_name, f"model1_frame_{frame_id:06d}.jpg"), img_model1)
    cv2.imwrite(os.path.join(vis_folder, video_name, f"model2_frame_{frame_id:06d}.jpg"), img_model2)
    cv2.imwrite(os.path.join(vis_folder, video_name, f"ensemble_frame_{frame_id:06d}.jpg"), img_ensemble)

def visualize_gbi_tracks(dataset_path, gbi_folder, vis_folder_gbi):
    """Visualize tracked objects from GBI output, saving to disk."""
    for file_name in os.listdir(gbi_folder):
        video_name = file_name.split('.')[0]
        gbi_path = os.path.join(gbi_folder, file_name)
        
        # Read GBI results (format: frame_id, track_id, x, y, w, h, conf, ...)
        tracks = {}
        with open(gbi_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                if frame_id not in tracks:
                    tracks[frame_id] = []
                tracks[frame_id].append((track_id, x, y, w, h))
        
        # Use dataset_path directly as the image directory
        video_img_path = dataset_path
        if not os.path.exists(video_img_path):
            print(f"Warning: Image path {video_img_path} not found for GBI visualization")
            continue
        
        img_paths = sorted([os.path.join(video_img_path, img) for img in os.listdir(video_img_path) if img.endswith(('.jpg', '.png'))])
        
        # Visualize each frame
        for frame_id in tracks:
            # Find the corresponding image
            img_idx = frame_id - 1  # Frame IDs are 1-based, image list is 0-based
            if img_idx >= len(img_paths):
                print(f"Warning: No image found for frame {frame_id} in {video_name}")
                continue
            
            np_img = cv2.imread(img_paths[img_idx])
            if np_img is None:
                print(f"Warning: Failed to load image {img_paths[img_idx]}")
                continue
            
            # Draw tracks in yellow with track IDs
            for track_id, x, y, w, h in tracks[frame_id]:
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                cv2.rectangle(np_img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow
                cv2.putText(np_img, f"ID:{track_id}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Save visualization
            vis_path = os.path.join(vis_folder_gbi, video_name)
            os.makedirs(vis_path, exist_ok=True)
            cv2.imwrite(os.path.join(vis_path, f"gbi_frame_{frame_id:06d}.jpg"), np_img)

def visualize_selected_frames(dataset_path, gbi_folder, stored_detections, selected_frames, video_name, selected_folder):
    """Visualize model1, model2, ensemble, and GBI tracks for 10 selected frames."""
    # Use dataset_path directly as the image directory
    video_img_path = dataset_path
    if not os.path.exists(video_img_path):
        print(f"Warning: Image path {video_img_path} not found for selected frames visualization")
        return
    
    img_paths = sorted([os.path.join(video_img_path, img) for img in os.listdir(video_img_path) if img.endswith(('.jpg', '.png'))])
    
    # Read GBI results
    gbi_path = os.path.join(gbi_folder, f"{video_name}.txt")
    tracks = {}
    if os.path.exists(gbi_path):
        with open(gbi_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                if frame_id not in tracks:
                    tracks[frame_id] = []
                tracks[frame_id].append((track_id, x, y, w, h))
    
    # Visualize selected frames
    for frame_id in selected_frames:
        # Load image
        img_idx = frame_id - 1  # Frame IDs are 1-based, image list is 0-based
        if img_idx >= len(img_paths):
            print(f"Warning: No image found for frame {frame_id} in {video_name}")
            continue
        
        np_img = cv2.imread(img_paths[img_idx])
        if np_img is None:
            print(f"Warning: Failed to load image {img_paths[img_idx]}")
            continue
        
        # Visualize model1 detections (blue)
        img_model1 = np_img.copy()
        if frame_id in stored_detections['model1']:
            for x1, y1, x2, y2, conf in stored_detections['model1'][frame_id]:
                cv2.rectangle(img_model1, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(img_model1, f"{conf:.2f}", (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Visualize model2 detections (green)
        img_model2 = np_img.copy()
        if frame_id in stored_detections['model2']:
            for x1, y1, x2, y2, conf in stored_detections['model2'][frame_id]:
                cv2.rectangle(img_model2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_model2, f"{conf:.2f}", (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Visualize ensemble detections (red)
        img_ensemble = np_img.copy()
        if frame_id in stored_detections['ensemble']:
            for x1, y1, x2, y2, conf in stored_detections['ensemble'][frame_id]:
                cv2.rectangle(img_ensemble, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img_ensemble, f"{conf:.2f}", (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Visualize GBI tracks (yellow)
        img_gbi = np_img.copy()
        if frame_id in tracks:
            for track_id, x, y, w, h in tracks[frame_id]:
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                cv2.rectangle(img_gbi, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img_gbi, f"ID:{track_id}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Save visualizations
        for subfolder, img in [('model1', img_model1), ('model2', img_model2), ('ensemble', img_ensemble), ('gbi', img_gbi)]:
            vis_path = os.path.join(selected_folder, video_name, subfolder)
            os.makedirs(vis_path, exist_ok=True)
            cv2.imwrite(os.path.join(vis_path, f"frame_{frame_id:06d}.jpg"), img)

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
    stored_detections = {'model1': {}, 'model2': {}, 'ensemble': {}}
    first_video_name = None
    frame_ids = []

    model1 = YoloDetector(args.model1_path)
    model2 = YoloDetector(args.model2_path)
    det = EnsembleDetector(model1, model2, args.model1_weight, args.model2_weight, args.iou_thresh, args.conf_thresh)
    
    # Set up visualization folder
    vis_folder = os.path.join(args.result_folder, args.exp_name, "visualizations") if args.visualize else None
    if args.visualize:
        os.makedirs(vis_folder, exist_ok=True)
    
    for (img, np_img), _, info, img_path in my_data_loader(args.dataset_path, args.track_percent):
        frame_id = info[2].item()
        video_name = info[4][0].split("/")[0]
        tag = f"{video_name}:{frame_id}"
        if video_name not in results:
            results[video_name] = []
            if first_video_name is None:
                first_video_name = video_name

        print(f"Processing {video_name}:{frame_id}\r", end="")
        if frame_id == 1:
            print(f"Initializing tracker for {video_name}")
            print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
            if tracker is not None:
                tracker.dump_cache()
            tracker = BoostTrack(video_name=video_name)

        # Get individual and ensemble predictions
        model1_preds = model1(np_img)
        model2_preds = model2(np_img)
        ensemble_preds = det(np_img)
        
        # Store detections for the first video
        if args.visualize and video_name == first_video_name:
            stored_detections['model1'][frame_id] = model1_preds.tolist() if len(model1_preds) > 0 else []
            stored_detections['model2'][frame_id] = model2_preds.tolist() if len(model2_preds) > 0 else []
            stored_detections['ensemble'][frame_id] = ensemble_preds.tolist() if len(ensemble_preds) > 0 else []
            frame_ids.append(frame_id)
        
        # Visualize if enabled
        if args.visualize:
            visualize_detections(np_img, model1_preds, model2_preds, ensemble_preds, video_name, frame_id, vis_folder)
        
        start_time = time.time()
        if ensemble_preds is None:
            continue
        targets = tracker.update(ensemble_preds, img, np_img, tag)
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
    if args.visualize:
        print(f"Visualizations saved to {vis_folder}")

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
        vis_folder_gbi = os.path.join(args.result_folder, args.exp_name + "_post_gbi", "visualizations") if args.visualize else None
        if args.visualize:
            os.makedirs(vis_folder_gbi, exist_ok=True)
        
        if not os.path.exists(post_folder_gbi):
            os.makedirs(post_folder_gbi)
        for file_name in os.listdir(post_folder_data):
            in_path = os.path.join(post_folder_data, file_name)
            out_path2 = os.path.join(post_folder_gbi, file_name)
            GBInterpolation(path_in=in_path, path_out=out_path2, interval=args.interval)
        print(f"Gradient boosting interpolation post-processing applied, saved to {post_folder_gbi}.")
        
        # Visualize GBI tracks
        if args.visualize:
            visualize_gbi_tracks(args.dataset_path, post_folder_gbi, vis_folder_gbi)
            print(f"GBI visualizations saved to {vis_folder_gbi}")
        
        # Visualize 10 selected frames for the first video
        if args.visualize and first_video_name:
            selected_folder = os.path.join(args.result_folder, args.exp_name + "_post_gbi", "selected_frames")
            os.makedirs(selected_folder, exist_ok=True)
            # Select 10 evenly spaced frames
            if len(frame_ids) >= 10:
                step = len(frame_ids) // 10
                selected_frames = [frame_ids[i * step] for i in range(10)]
            else:
                selected_frames = frame_ids[:10]  # Use all if fewer than 10
            visualize_selected_frames(args.dataset_path, post_folder_gbi, stored_detections, selected_frames, first_video_name, selected_folder)
            print(f"Selected frame visualizations saved to {selected_folder}")

if __name__ == "__main__":
    main()
