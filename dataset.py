import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from tracker.assoc import iou_batch
from tracker.boost_track import KalmanBoxTracker, convert_bbox_to_z
from tracker.embedding import EmbeddingComputer
from torch_geometric.data import Data
from detectors import EnsembleDetector

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
        print(f"Initializing MOTDataset: data_dir={data_dir}, json_file={json_file}, img_size={img_size}")
        self.input_dim = img_size
        self.data_dir = data_dir
        self.img_data_dir = img_data_dir or data_dir
        self.json_file = json_file
        self.coco = COCO(os.path.join(self.data_dir, self.json_file))
        self.ids = self.coco.getImgIds()
        print(f"Loaded {len(self.ids)} image IDs from COCO")
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        print(f"Classes: {self._classes}")
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        print("Loading COCO annotations...")
        annotations = [self.load_anno_from_ids(_ids) for _ids in self.ids]
        print(f"Loaded {len(annotations)} annotations")
        return annotations

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        print(f"Loading annotations for image ID {id_}: frame_id={frame_id}, video_id={video_id}")
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
        if len(objs) > 0:
            print(f"Image ID {id_}: {len(objs)} valid objects, first object: bbox={objs[0]['clean_bbox']}, track_id={objs[0]['track_id']}")

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
        print(f"Loading image: {img_file}")
        img = cv2.imread(img_file)
        assert img is not None, f"Image {img_file} not found"
        print(f"Image {img_file} loaded, shape: {img.shape}")
        return img, res.copy(), img_info, np.array([id_])

    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        print(f"Preprocessing image {img_id}, target shape: {target.shape}")
        tensor, target = self.preproc(img, target, self.input_dim)
        print(f"Preprocessed tensor shape: {tensor.shape}")
        return (tensor, img), target, img_info, img_id

class ValTransform:
    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1)):
        print(f"Initializing ValTransform: means={rgb_means}, std={std}, swap={swap}")
        self.means = rgb_means
        self.swap = swap
        self.std = std

    def __call__(self, img, res, input_size):
        print(f"Applying ValTransform: input_size={input_size}, img shape={img.shape}")
        img, _ = preproc(img, input_size, self.means, self.std, self.swap)
        print(f"Transformed img shape: {img.shape}")
        return img, np.zeros((1, 5))

def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    print(f"Preprocessing image: input_size={input_size}, mean={mean}, std={std}")
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    print(f"Resize ratio: {r}")
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
    print(f"Preprocessed image shape: {padded_img.shape}")
    return padded_img, r

class MOTGraphDataset(MOTDataset):
    def __init__(self, *args, model1=None, model2=None, reid_path=None, **kwargs):
        print(f"Initializing MOTGraphDataset: reid={reid_path}")
        super().__init__(*args, **kwargs)
        self.detector = EnsembleDetector(
            model1,
            model2,
            model1_weight=kwargs.get('model1_weight', 0.4),
            model2_weight=kwargs.get('model2_weight', 0.6),
            iou_thresh=kwargs.get('iou_thresh', 0.6),
            conf_thresh=kwargs.get('conf_thresh', 0.3)
        )
        print("EnsembleDetector initialized.")
        self.embedder = EmbeddingComputer(
            dataset=kwargs.get('dataset', 'mot20'),
            test_dataset=kwargs.get('test_dataset', False),
            grid_off=True,
            reid_path=reid_path
        )
        print("EmbeddingComputer initialized.")
        self.trackers = {}

    def __getitem__(self, index):
        print(f"Fetching item {index}")
        img, target, img_info, img_id = super().__getitem__(index)
        height, width, frame_id, video_id, file_name = img_info
        print(f"Item {index}: height={height}, width={width}, frame_id={frame_id}, video_id={video_id}, file_name={file_name}")
        img_numpy = img[1]
        print(f"Image shape: {img_numpy.shape}")
        dets = self.detector(img_numpy)
        if dets is None:
            dets = np.zeros((0, 5))
        print(f"Detections shape: {dets.shape}")
        dets = dets[dets[:, 4] >= 0.4]
        print(f"Filtered detections (>=0.4 conf): {dets.shape}")
        dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], f"{video_id}:{frame_id}") if len(dets) > 0 else np.zeros((0, 256))
        print(f"Embeddings shape: {dets_embs.shape}")
        if video_id not in self.trackers:
            self.trackers[video_id] = []
        trackers = self.trackers[video_id]
        print(f"Trackers for video {video_id}: {len(trackers)}")
        trk_states = []
        trk_embs = []
        matched, unmatched_dets, unmatched_trks = self._associate_gt(dets, trackers, target[:, :4], target[:, 5])
        print(f"Associations: {len(matched)} matched, {len(unmatched_dets)} unmatched dets, {len(unmatched_trks)} unmatched tracks")
        first_match = True
        for m in matched:
            det_idx, trk_idx = m
            trackers[trk_idx].update(dets[det_idx, :4], dets[det_idx, 4])
            trackers[trk_idx].update_emb(dets_embs[det_idx], alpha=0.95)
            if first_match:
                print(f"First matched update: tracker {trk_idx} with det {det_idx}")
                first_match = False
        first_unmatched_det = True
        for i in unmatched_dets:
            if dets[i, 4] >= 0.4:
                tracker = KalmanBoxTracker(dets[i, :], emb=dets_embs[i])
                tracker.gt_id = self._get_det_gt_id(dets[i, :4], target[:, :4], target[:, 5])
                trackers.append(tracker)
                if first_unmatched_det:
                    print(f"First new tracker for det {i}, gt_id={tracker.gt_id}")
                    first_unmatched_det = False
        first_tracker = True
        for t in trackers:
            t.predict()
            trk_states.append(t.get_state()[0])
            trk_embs.append(t.get_emb())
            if first_tracker:
                print(f"First tracker predicted state: {trk_states[-1]}")
                first_tracker = False
        trk_states = np.array(trk_states) if trk_states else np.zeros((0, 4))
        trk_embs = np.array(trk_embs) if trk_embs else np.zeros((0, 256))
        print(f"Tracker states shape: {trk_states.shape}, embeddings shape: {trk_embs.shape}")
        self.trackers[video_id] = [t for t in trackers if t.time_since_update <= 100]
        print(f"Filtered trackers for video {video_id}: {len(self.trackers[video_id])}")
        graph = self._create_graph(dets, dets_embs, trk_states, trk_embs, trackers, target[:, :4], target[:, 5], width, height)
        print(f"Graph created: nodes={graph.x.shape[0]}, edges={graph.edge_index.shape[1]}")
        return graph

    def _associate_gt(self, dets, trackers, gt_boxes, gt_ids):
        print(f"Associating: {len(dets)} dets, {len(trackers)} trackers, {len(gt_boxes)} gt boxes")
        if len(trackers) == 0 or len(dets) == 0:
            print("No trackers or detections, returning empty matches")
            return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.arange(len(trackers))
        trk_states = np.array([t.get_state()[0] for t in trackers])
        print(f"Tracker states shape: {trk_states.shape}")
        iou_matrix = iou_batch(dets, trk_states)
        print(f"IOU matrix shape: {iou_matrix.shape}")
        cost_matrix = 1 - iou_matrix
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched = [[i, j] for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] >= 0.35]
        unmatched_dets = [i for i in range(len(dets)) if i not in row_ind]
        unmatched_trks = [j for j in range(len(trackers)) if j not in col_ind]
        print(f"Matched: {len(matched)}, Unmatched dets: {len(unmatched_dets)}, Unmatched tracks: {len(unmatched_trks)}")
        return np.array(matched), np.array(unmatched_dets), np.array(unmatched_trks)

    def _create_graph(self, dets, dets_embs, trk_states, trk_embs, trackers, gt_boxes, gt_ids, img_w, img_h):
        print(f"Creating graph: {len(dets)} dets, {len(trk_states)} trackers, img_w={img_w}, img_h={img_h}")
        node_features = []
        first_det = True
        for i, det in enumerate(dets):
            x1, y1, x2, y2 = det[:4] / np.array([img_w, img_h, img_w, img_h])
            emb = dets_embs[i]
            node_features.append(np.concatenate([[x1, y1, x2, y2], emb]))
            if first_det:
                print(f"First det node: bbox=[{x1}, {y1}, {x2}, {y2}], emb shape={emb.shape}")
                first_det = False
        first_trk = True
        for i, trk in enumerate(trk_states):
            x1, y1, x2, y2 = trk / np.array([img_w, img_h, img_w, img_h])
            emb = trk_embs[i]
            node_features.append(np.concatenate([[x1, y1, x2, y2], emb]))
            if first_trk:
                print(f"First track node: bbox=[{x1}, {y1}, {x2}, {y2}], emb shape={emb.shape}")
                first_trk = False
        node_features = torch.tensor(node_features, dtype=torch.float32) if node_features else torch.zeros((0, 260))
        print(f"Node features shape: {node_features.shape}")
        edge_index = []
        edge_labels = []
        edge_attr = []
        first_edge = True
        for i in range(len(dets)):
            for j in range(len(trk_states)):
                iou = iou_batch(dets[i:i+1, :4], trk_states[j:j+1])[0, 0]
                if iou > 0.1:
                    edge_index.append([i, j + len(dets)])
                    sim = np.dot(dets_embs[i], trk_embs[j]) / (np.linalg.norm(dets_embs[i]) * np.linalg.norm(trk_embs[j]) + 1e-6)
                    mhd = self._compute_mhd(dets[i, :4], trk_states[j], trackers[j].kf)
                    edge_attr.append([iou, sim, mhd])
                    det_gt_id = self._get_det_gt_id(dets[i, :4], gt_boxes, gt_ids)
                    trk_gt_id = trackers[j].gt_id if hasattr(trackers[j], 'gt_id') else -1
                    edge_labels.append(1 if det_gt_id == trk_gt_id and det_gt_id != -1 else 0)
                    if first_edge:
                        print(f"First edge: iou={iou}, sim={sim}, mhd={mhd}, label={edge_labels[-1]}")
                        first_edge = False
        edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.zeros((2, 0), dtype=torch.long)
        edge_labels = torch.tensor(edge_labels, dtype=torch.float32) if edge_labels else torch.zeros((0,), dtype=torch.float32)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.zeros((0, 3), dtype=torch.float32)
        print(f"Graph: {edge_index.shape[1]} edges, edge_attr shape={edge_attr.shape}, labels shape={edge_labels.shape}")
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)

    def _compute_mhd(self, det, trk, kf):
        print(f"Computing MHD: det={det}, trk={trk}")
        z = convert_bbox_to_z(det).reshape(-1)
        x = kf.x[:4]
        sigma_inv = np.reciprocal(np.diag(kf.covariance[:4, :4]))
        mhd = ((z - x) ** 2 * sigma_inv).sum()
        normalized_mhd = min(mhd, 13.2767) / 13.2767
        print(f"MHD: raw={mhd}, normalized={normalized_mhd}")
        return normalized_mhd

    def _get_det_gt_id(self, det, gt_boxes, gt_ids):
        print(f"Computing GT ID for det: {det}")
        ious = iou_batch(det.reshape(1, -1), gt_boxes)
        max_iou = ious.max()
        if max_iou >= 0.5:
            gt_id = gt_ids[np.argmax(ious)]
            print(f"Matched det to GT ID {gt_id} with IOU {max_iou}")
            return gt_id
        print("No GT match (IOU < 0.5)")
        return -1