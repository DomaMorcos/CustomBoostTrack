"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import os
from copy import deepcopy
from typing import Optional, List

import cv2
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.optimize import linear_sum_assignment
from loguru import logger

from default_settings import GeneralSettings, BoostTrackSettings, BoostTrackPlusPlusSettings
from tracker.embedding import EmbeddingComputer
from tracker.assoc import associate, iou_batch, MhDist_similarity, shape_similarity, soft_biou_batch
from tracker.ecc import ECC
from tracker.kalmanfilter import KalmanFilter
from tracker.GNN import MOTGNN

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,h,r] where x,y is the centre of the box and h is the height and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    r = w / float(h + 1e-6)
    return np.array([x, y, h, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,h,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    h = x[2]
    r = x[3]
    w = 0 if r <= 0 else r * h
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, emb: Optional[np.ndarray] = None):
        """
        Initialises a tracker using initial bounding box.
        """
        self.bbox_to_z_func = convert_bbox_to_z
        self.x_to_bbox_func = convert_x_to_bbox
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.kf = KalmanFilter(self.bbox_to_z_func(bbox))
        self.emb = emb
        self.hit_streak = 0
        self.age = 0

    def get_confidence(self, coef: float = 0.9) -> float:
        n = 7
        if self.age < n:
            return coef ** (n - self.age)
        return coef ** (self.time_since_update-1)

    def update(self, bbox: np.ndarray, score: float = 0):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z_func(bbox), score)

    def camera_update(self, transform: np.ndarray):
        x1, y1, x2, y2 = self.get_state()[0]
        x1_, y1_, _ = transform @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = transform @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:4] = [cx, cy, h, w / h]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

class BoostTrack(object):
    def __init__(self, video_name: Optional[str] = None, gnn_model_path: Optional[str] = None):
        self.frame_count = 0
        self.trackers: List[KalmanBoxTracker] = []
        self.max_age = GeneralSettings.max_age(video_name)
        self.iou_threshold = GeneralSettings['iou_threshold']
        self.det_thresh = GeneralSettings['det_thresh']
        self.min_hits = GeneralSettings['min_hits']
        self.lambda_iou = BoostTrackSettings['lambda_iou']
        self.lambda_mhd = BoostTrackSettings['lambda_mhd']
        self.lambda_shape = BoostTrackSettings['lambda_shape']
        self.use_dlo_boost = BoostTrackSettings['use_dlo_boost']
        self.use_duo_boost = BoostTrackSettings['use_duo_boost']
        self.dlo_boost_coef = BoostTrackSettings['dlo_boost_coef']
        self.use_rich_s = BoostTrackPlusPlusSettings['use_rich_s']
        self.use_sb = BoostTrackPlusPlusSettings['use_sb']
        self.use_vt = BoostTrackPlusPlusSettings['use_vt']
        if GeneralSettings['use_embedding']:
            self.embedder = EmbeddingComputer(GeneralSettings['dataset'], GeneralSettings['test_dataset'], True, reid_path=GeneralSettings['reid_path'])
        else:
            self.embedder = None
        if GeneralSettings['use_ecc']:
            self.ecc = ECC(scale=350, video_name=video_name, use_cache=True)
        else:
            self.ecc = None

        # GNN initialization
        self.gnn = None
        if gnn_model_path:
            self.gnn = MOTGNN(input_dim=260, hidden_dim=128, edge_dim=3).to('cuda')
            self.gnn.load_state_dict(torch.load(gnn_model_path))
            self.gnn.eval()
            logger.info(f"Loaded GNN model from {gnn_model_path}")

    def update(self, dets, img_tensor, img_numpy, tag):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if dets is None:
            return np.empty((0, 5))
        if not isinstance(dets, np.ndarray):
            dets = dets.cpu().detach().numpy()

        self.frame_count += 1

        # Rescale
        scale = min(img_tensor.shape[2] / img_numpy.shape[0], img_tensor.shape[3] / img_numpy.shape[1])
        dets = deepcopy(dets)
        dets[:, :4] /= scale

        if self.ecc is not None:
            transform = self.ecc(img_numpy, self.frame_count, tag)
            for trk in self.trackers:
                trk.camera_update(transform)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        confs = np.zeros((len(self.trackers), 1))

        for t in range(len(trks)):
            pos = self.trackers[t].predict()[0]
            confs[t] = self.trackers[t].get_confidence()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], confs[t, 0]]

        if self.use_dlo_boost:
            dets = self.dlo_confidence_boost(dets, self.use_rich_s, self.use_sb, self.use_vt)

        if self.use_duo_boost:
            dets = self.duo_confidence_boost(dets)

        remain_inds = dets[:, 4] >= self.det_thresh
        dets = dets[remain_inds]
        scores = dets[:, 4]

        # Generate embeddings
        dets_embs = np.ones((dets.shape[0], 1))
        emb_cost = None
        if self.embedder and dets.size > 0:
            dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], tag)
            trk_embs = []
            for t in range(len(self.trackers)):
                trk_embs.append(self.trackers[t].get_emb())
            trk_embs = np.array(trk_embs)
            if trk_embs.size > 0 and dets.size > 0:
                emb_cost = dets_embs.reshape(dets_embs.shape[0], -1) @ trk_embs.reshape((trk_embs.shape[0], -1)).T
        emb_cost = None if self.embedder is None else emb_cost

        # GNN-based association
        if self.gnn and len(dets) > 0 and len(self.trackers) > 0:
            graph = self._create_inference_graph(dets, dets_embs, trks, trk_embs, img_numpy.shape[:2])
            with torch.no_grad():
                edge_scores = self.gnn(graph.x.cuda(), graph.edge_index.cuda(), graph.edge_attr.cuda()).cpu().numpy()
            cost_matrix = 1 - edge_scores.reshape(len(dets), len(self.trackers))
            matched, unmatched_dets, unmatched_trks = self._hungarian_matching(cost_matrix)
            sym_matrix = cost_matrix
        else:
            matched, unmatched_dets, unmatched_trks, sym_matrix = associate(
                dets,
                trks,
                self.iou_threshold,
                mahalanobis_distance=self.get_mh_dist_matrix(dets),
                track_confidence=confs,
                detection_confidence=scores,
                emb_cost=emb_cost,
                lambda_iou=self.lambda_iou,
                lambda_mhd=self.lambda_mhd,
                lambda_shape=self.lambda_shape
            )

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = 0.95
        dets_alpha = af + (1 - af) * (1 - trust)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], scores[m[0]])
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        for i in unmatched_dets:
            if dets[i, 4] >= self.det_thresh:
                self.trackers.append(KalmanBoxTracker(dets[i, :], emb=dets_embs[i]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1], [trk.get_confidence()])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def dump_cache(self):
        if self.ecc is not None:
            self.ecc.save_cache()

    def get_iou_matrix(self, detections: np.ndarray, buffered: bool = False) -> np.ndarray:
        trackers = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], self.trackers[t].get_confidence()]
        return iou_batch(detections, trackers) if not buffered else soft_biou_batch(detections, trackers)

    def get_mh_dist_matrix(self, detections: np.ndarray, n_dims: int = 4) -> np.ndarray:
        if len(self.trackers) == 0:
            return np.zeros((0, 0))
        z = np.zeros((len(detections), n_dims), dtype=float)
        x = np.zeros((len(self.trackers), n_dims), dtype=float)
        sigma_inv = np.zeros_like(x, dtype=float)
        f = self.trackers[0].bbox_to_z_func
        for i in range(len(detections)):
            z[i, :n_dims] = f(detections[i, :]).reshape((-1, ))[:n_dims]
        for i in range(len(self.trackers)):
            x[i] = self.trackers[i].kf.x[:n_dims]
            sigma_inv[i] = np.reciprocal(np.diag(self.trackers[i].kf.covariance[:n_dims, :n_dims]))
        return ((z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2 * sigma_inv.reshape((1, -1, n_dims))).sum(axis=2)

    def duo_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        n_dims = 4
        limit = 13.2767
        mahalanobis_distance = self.get_mh_dist_matrix(detections, n_dims)
        if mahalanobis_distance.size > 0 and self.frame_count > 1:
            min_mh_dists = mahalanobis_distance.min(1)
            mask = (min_mh_dists > limit) & (detections[:, 4] < self.det_thresh)
            boost_detections = detections[mask]
            boost_detections_args = np.argwhere(mask).reshape((-1,))
            iou_limit = 0.3
            if len(boost_detections) > 0:
                bdiou = iou_batch(boost_detections, boost_detections) - np.eye(len(boost_detections))
                bdiou_max = bdiou.max(axis=1)
                remaining_boxes = boost_detections_args[bdiou_max <= iou_limit]
                args = np.argwhere(bdiou_max > iou_limit).reshape((-1,))
                for i in range(len(args)):
                    boxi = args[i]
                    tmp = np.argwhere(bdiou[boxi] > iou_limit).reshape((-1,))
                    args_tmp = np.append(np.intersect1d(boost_detections_args[args], boost_detections_args[tmp]), boost_detections_args[boxi])
                    conf_max = np.max(detections[args_tmp, 4])
                    if detections[boost_detections_args[boxi], 4] == conf_max:
                        remaining_boxes = np.array(remaining_boxes.tolist() + [boost_detections_args[boxi]])
                mask = np.zeros_like(detections[:, 4], dtype=np.bool_)
                mask[remaining_boxes] = True
            detections[:, 4] = np.where(mask, self.det_thresh + 1e-4, detections[:, 4])
        return detections

    def dlo_confidence_boost(self, detections: np.ndarray, use_rich_sim: bool, use_soft_boost: bool, use_varying_th: bool) -> np.ndarray:
        sbiou_matrix = self.get_iou_matrix(detections, True)
        if sbiou_matrix.size == 0:
            return detections
        trackers = np.zeros((len(self.trackers), 6))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, self.trackers[t].time_since_update - 1]
        if use_rich_sim:
            mhd_sim = MhDist_similarity(self.get_mh_dist_matrix(detections), 1)
            shape_sim = shape_similarity(detections, trackers)
            S = (mhd_sim + shape_sim + sbiou_matrix) / 3
        else:
            S = self.get_iou_matrix(detections, False)
        if not use_soft_boost and not use_varying_th:
            max_s = S.max(1)
            coef = self.dlo_boost_coef
            detections[:, 4] = np.maximum(detections[:, 4], max_s * coef)
        else:
            if use_soft_boost:
                max_s = S.max(1)
                alpha = 0.65
                detections[:, 4] = np.maximum(detections[:, 4], alpha*detections[:, 4] + (1-alpha)*max_s**(1.5))
            if use_varying_th:
                threshold_s = 0.95
                threshold_e = 0.8
                n_steps = 20
                alpha = (threshold_s - threshold_e) / n_steps
                tmp = (S > np.maximum(threshold_s - trackers[:, 5] * alpha, threshold_e)).max(1)
                scores = deepcopy(detections[:, 4])
                scores[tmp] = np.maximum(scores[tmp], self.det_thresh + 1e-5)
                detections[:, 4] = scores
        return detections

    def _create_inference_graph(self, dets, dets_embs, trks, trk_embs, img_shape):
        h, w = img_shape
        node_features = []
        for i, det in enumerate(dets):
            x1, y1, x2, y2 = det[:4] / np.array([w, h, w, h])
            emb = dets_embs[i]
            node_features.append(np.concatenate([[x1, y1, x2, y2], emb]))
        for i, trk in enumerate(trks):
            x1, y1, x2, y2 = trk[:4] / np.array([w, h, w, h])
            emb = trk_embs[i]
            node_features.append(np.concatenate([[x1, y1, x2, y2], emb]))
        node_features = torch.tensor(node_features, dtype=torch.float32)

        edge_index = []
        edge_attr = []
        for i in range(len(dets)):
            for j in range(len(trks)):
                iou = iou_batch(dets[i:i+1, :4], trks[j:j+1, :4])[0, 0]
                if iou > 0.1:
                    edge_index.append([i, j + len(dets)])
                    sim = np.dot(dets_embs[i], trk_embs[j]) / (np.linalg.norm(dets_embs[i]) * np.linalg.norm(trk_embs[j]) + 1e-6)
                    mhd = self._compute_mhd(dets[i, :4], trks[j], self.trackers[j].kf)
                    edge_attr.append([iou, sim, mhd])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.zeros((0, 3), dtype=torch.float32)

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    def _compute_mhd(self, det, trk, kf):
        z = convert_bbox_to_z(det).reshape(-1)
        x = kf.x[:4]
        sigma_inv = np.reciprocal(np.diag(kf.covariance[:4, :4]))
        mhd = ((z - x) ** 2 * sigma_inv).sum()
        return min(mhd, 13.2767) / 13.2767

    def _hungarian_matching(self, cost_matrix):
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched = [[i, j] for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] < 1 - self.iou_threshold]
        unmatched_dets = [i for i in range(len(cost_matrix)) if i not in row_ind]
        unmatched_trks = [j for j in range(cost_matrix.shape[1]) if j not in col_ind]
        return matched, unmatched_dets, unmatched_trks