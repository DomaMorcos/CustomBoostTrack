import os
import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from tracker.assoc import iou_batch
from tracker.boost_track import KalmanBoxTracker, convert_bbox_to_z
from tracker.embedding import EmbeddingComputer
from torch_geometric.data import Data
from detectors import EnsembleDetector
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MOTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        json_file="val_half.json",
        name="train",
        img_size=(960, 1728),
        preproc=None,
        img_data_dir=None
    ):
        self.input_dim = img_size
        self.data_dir = data_dir
        self.img_data_dir = img_data_dir or data_dir
        self.json_file = json_file
        self.coco = COCO(os.path.join(self.data_dir, self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        annotations = [self.load_anno_from_ids(_ids) for _ids in self.ids]
        return annotations

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)
        num_objs = len(objs)
        res = np.zeros((num_objs, 6))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]
        file_name = im_ann["file_name"]
        img_info = (height, width, frame_id, video_id, file_name)
        del im_ann, annotations
        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]
        res, img_info, file_name = self.annotations[index]
        img_file = os.path.join(self.img_data_dir, file_name)
        img = cv2.imread(img_file)
        assert img is not None, f"Image {img_file} not found"
        return img, res.copy(), img_info, np.array([id_])

    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        tensor, target = self.preproc(img, target, self.input_dim)
        return (tensor, img), target, img_info, img_id

class ValTransform:
    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1), is_train=False):
        self.means = rgb_means
        self.swap = swap
        self.std = std
        self.is_train = is_train
        self.aug_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.3, blur_limit=(3, 7)),
            A.HueSaturationValue(p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __call__(self, img, res, input_size):
        if self.is_train and len(res) > 0:
            img_h, img_w = input_size[0], input_size[1]
            bboxes = res[:, :4].copy()
            # Filter valid bboxes (x_max > x_min, y_max > y_min)
            valid_mask = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
            bboxes = bboxes[valid_mask]
            labels = res[valid_mask, 4].tolist()
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]] / img_w, 0.0, 1.0)  # x1, x2
                bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]] / img_h, 0.0, 1.0)  # y1, y2
                bboxes = bboxes.tolist()
                aug_result = self.aug_transform(image=img, bboxes=bboxes, labels=labels)
                img = aug_result['image']
                if len(aug_result['bboxes']) > 0:
                    aug_bboxes = np.array(aug_result['bboxes'])
                    # Filter valid augmented bboxes with minimum size
                    valid_aug_mask = (aug_bboxes[:, 2] > aug_bboxes[:, 0] + 1e-6) & (aug_bboxes[:, 3] > aug_bboxes[:, 1] + 1e-6)
                    aug_bboxes = aug_bboxes[valid_aug_mask]
                    aug_labels = np.array(aug_result['labels'])[valid_aug_mask].tolist()
                    if len(aug_bboxes) > 0:
                        aug_bboxes[:, [0, 2]] = np.clip(aug_bboxes[:, [0, 2]], 0.0, 1.0)
                        aug_bboxes[:, [1, 3]] = np.clip(aug_bboxes[:, [1, 3]], 0.0, 1.0)
                        aug_bboxes[:, [0, 2]] *= img_w
                        aug_bboxes[:, [1, 3]] *= img_h
                        # Update res with valid augmented bboxes
                        new_res = np.zeros((len(aug_bboxes), 6))
                        new_res[:, :4] = aug_bboxes
                        new_res[:, 4] = aug_labels
                        new_res[:, 5] = res[valid_mask][:len(aug_bboxes), 5]  # Preserve track IDs
                        res = new_res
                    else:
                        # No valid augmented bboxes
                        aug_result = self.aug_transform(image=img, bboxes=[], labels=[])
                        img = aug_result['image']
                        res = np.zeros((0, 6))
                else:
                    # No bboxes after augmentation
                    aug_result = self.aug_transform(image=img, bboxes=[], labels=[])
                    img = aug_result['image']
                    res = np.zeros((0, 6))
            else:
                # No valid input bboxes
                aug_result = self.aug_transform(image=img, bboxes=[], labels=[])
                img = aug_result['image']
                res = np.zeros((0, 6))
        img, _ = preproc(img, input_size, self.means, self.std, self.swap)
        return img, np.zeros((1, 5)) if not self.is_train else (img, res)

def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

class MOTGraphDataset(MOTDataset):
    def __init__(self, *args, model1=None, model2=None, reid_path=None, **kwargs):
        mot_kwargs = {
            'data_dir': kwargs.get('data_dir'),
            'json_file': kwargs.get('json_file', 'val_half.json'),
            'name': kwargs.get('name', 'train'),
            'img_size': kwargs.get('img_size', (960, 1728)),
            'preproc': kwargs.get('preproc'),
            'img_data_dir': kwargs.get('img_data_dir')
        }
        super().__init__(**mot_kwargs)
        self.detector = EnsembleDetector(
            model1,
            model2,
            model1_weight=kwargs.get('model1_weight', 0.4),
            model2_weight=kwargs.get('model2_weight', 0.6),
            iou_thresh=kwargs.get('iou_thresh', 0.6),
            conf_thresh=kwargs.get('conf_thresh', 0.3)
        )
        self.embedder = EmbeddingComputer(
            dataset=kwargs.get('dataset', 'mot20'),
            test_dataset=kwargs.get('test_dataset', False),
            grid_off=True,
            reid_path=reid_path
        )
        self.trackers = {}
        self.prev_trackers = {}  # Store trackers from previous frame
        self.prev_frame_id = {}  # Track frame IDs per video

    def __getitem__(self, index):
        img, target, img_info, img_id = super().__getitem__(index)
        height, width, frame_id, video_id, file_name = img_info
        img_numpy = img[1]
        dets = self.detector(img_numpy)
        if dets is None:
            dets = np.zeros((0, 5))
        dets = torch.tensor(dets, dtype=torch.float32) if len(dets) > 0 else torch.zeros((0, 5))
        dets = dets[dets[:, 4] >= 0.4]
        dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4].cpu().numpy(), f"{video_id}:{frame_id}") if len(dets) > 0 else np.zeros((0, 256))
        dets_embs = torch.tensor(dets_embs, dtype=torch.float32)
        if video_id not in self.trackers:
            self.trackers[video_id] = []
        trackers = self.trackers[video_id]
        trk_states = []
        trk_embs = []
        matched, unmatched_dets, unmatched_trks = self._associate_gt(dets, trackers, target[:, :4], target[:, 5])
        for m in matched:
            det_idx, trk_idx = m
            trackers[trk_idx].update(dets[det_idx, :4], dets[det_idx, 4])
            trackers[trk_idx].update_emb(dets_embs[det_idx], alpha=0.95)
        for i in unmatched_dets:
            if dets[i, 4] >= 0.4:
                tracker = KalmanBoxTracker(dets[i, :], emb=dets_embs[i])
                tracker.gt_id = self._get_det_gt_id(dets[i, :4], target[:, :4], target[:, 5])
                trackers.append(tracker)
        for t in trackers:
            t.predict()
            trk_states.append(t.get_state()[0])
            trk_embs.append(t.get_emb())
        trk_states = np.array(trk_states) if trk_states else np.zeros((0, 4))
        trk_embs = np.array(trk_embs) if trk_embs else np.zeros((0, 256))
        self.trackers[video_id] = [t for t in trackers if t.time_since_update <= 100]
        # Store trackers for temporal modeling
        self.prev_trackers[video_id] = trackers.copy()
        self.prev_frame_id[video_id] = frame_id
        graph = self._create_graph(dets, dets_embs, trk_states, trk_embs, trackers, target[:, :4], target[:, 5], width, height, video_id, frame_id)
        return graph

    def _associate_gt(self, dets, trackers, gt_boxes, gt_ids):
        if len(trackers) == 0 or len(dets) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.arange(len(trackers))
        trk_states = np.array([t.get_state()[0] for t in trackers])
        iou_matrix = iou_batch(dets, trk_states)
        cost_matrix = 1 - iou_matrix
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched = [[i, j] for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] >= 0.35]
        unmatched_dets = [i for i in range(len(dets)) if i not in row_ind]
        unmatched_trks = [j for j in range(len(trackers)) if j not in col_ind]
        return np.array(matched), np.array(unmatched_dets), np.array(unmatched_trks)

    def _create_graph(self, dets, dets_embs, trk_states, trk_embs, trackers, gt_boxes, gt_ids, img_w, img_h, video_id, frame_id):
        node_features = []
        for i, det in enumerate(dets):
            x1, y1, x2, y2 = det[:4] / np.array([img_w, img_h, img_w, img_h])
            emb = dets_embs[i]
            node_features.append(np.concatenate([[x1, y1, x2, y2], emb]))
        for i, trk in enumerate(trk_states):
            x1, y1, x2, y2 = trk / np.array([img_w, img_h, img_w, img_h])
            emb = trk_embs[i]
            node_features.append(np.concatenate([[x1, y1, x2, y2], emb]))
        node_features = torch.tensor(node_features, dtype=torch.float32) if node_features else torch.zeros((0, 260))
        edge_index = []
        edge_labels = []
        edge_attr = []
        # Spatial edges (detections to trackers)
        for i in range(len(dets)):
            for j in range(len(trk_states)):
                iou = iou_batch(dets[i:i+1, :4], trk_states[j:j+1])[0, 0]
                if iou > 0.1:
                    edge_index.append([i, j + len(dets)])
                    sim = np.dot(dets_embs[i], trk_embs[j]) / (np.linalg.norm(dets_embs[i]) * np.linalg.norm(trk_embs[j]) + 1e-6)
                    mhd = self._compute_mhd(dets[i, :4], trk_states[j], trackers[j].kf)
                    edge_attr.append([iou, sim, mhd, 0.0])  # 0.0 for spatial edge
                    det_gt_id = self._get_det_gt_id(dets[i, :4], gt_boxes, gt_ids)
                    trk_gt_id = trackers[j].gt_id if hasattr(trackers[j], 'gt_id') else -1
                    edge_labels.append(1 if det_gt_id == trk_gt_id and det_gt_id != -1 else 0)
        # Temporal edges (trackers to previous trackers)
        if video_id in self.prev_trackers and video_id in self.prev_frame_id:
            prev_trackers = self.prev_trackers[video_id]
            prev_frame_id = self.prev_frame_id[video_id]
            frame_gap = frame_id - prev_frame_id
            if 1 <= frame_gap <= 2:
                prev_trk_states = np.array([t.get_state()[0] for t in prev_trackers]) if prev_trackers else np.zeros((0, 4))
                prev_trk_embs = np.array([t.get_emb() for t in prev_trackers]) if prev_trackers else np.zeros((0, 256))
                for j in range(len(trk_states)):
                    for k in range(len(prev_trk_states)):
                        iou = iou_batch(trk_states[j:j+1], prev_trk_states[k:k+1])[0, 0]
                        if iou > 0.1:
                            edge_index.append([j + len(dets), k + len(dets)])
                            sim = np.dot(trk_embs[j], prev_trk_embs[k]) / (np.linalg.norm(trk_embs[j]) * np.linalg.norm(prev_trk_embs[k]) + 1e-6)
                            mhd = self._compute_mhd(trk_states[j], prev_trk_states[k], trackers[j].kf)
                            edge_attr.append([iou, sim, mhd, frame_gap / 2.0])  # Normalize frame gap
                            trk_gt_id = trackers[j].gt_id if hasattr(trackers[j], 'gt_id') else -1
                            prev_trk_gt_id = prev_trackers[k].gt_id if hasattr(prev_trackers[k], 'gt_id') else -1
                            edge_labels.append(1 if trk_gt_id == prev_trk_gt_id and trk_gt_id != -1 else 0)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.zeros((2, 0), dtype=torch.long)
        edge_labels = torch.tensor(edge_labels, dtype=torch.float32) if edge_labels else torch.zeros((0,), dtype=torch.float32)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.zeros((0, 4), dtype=torch.float32)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)

    def _compute_mhd(self, det, trk, kf):
        z = convert_bbox_to_z(det).reshape(-1)
        x = kf.x[:4]
        sigma_inv = np.reciprocal(np.diag(kf.covariance[:4, :4]))
        mhd = ((z - x) ** 2 * sigma_inv).sum()
        normalized_mhd = min(mhd, 13.2767) / 13.2767
        return normalized_mhd

    def _get_det_gt_id(self, det, gt_boxes, gt_ids):
        ious = iou_batch(det.reshape(1, -1), gt_boxes)
        max_iou = ious.max()
        if max_iou >= 0.5:
            gt_id = gt_ids[np.argmax(ious)]
            return gt_id
        return -1