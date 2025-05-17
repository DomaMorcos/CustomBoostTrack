import torch
from ultralytics import YOLO
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from ensemble_boxes import weighted_boxes_fusion 



from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



from abc import ABC, abstractmethod

class Detector(ABC):
    @abstractmethod
    def __call__(self, img):
        pass

class YoloDetector(Detector):
    def __init__(self, yolo_path):
        self.model = YOLO(yolo_path)

    def __call__(self, img):
        results = self.model(img, half=True)[0]  # Let Ultralytics scale to input resolution
        annotations = []
        for box in results.boxes:
            if int(box.cls) == 0:  # Only keep 'person' class (class ID 0)
                xyxy = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]
                conf = box.conf[0].item()
                annotations.append(xyxy + [conf])
        return torch.tensor(annotations, dtype=torch.float32) if annotations else torch.zeros((0, 5), dtype=torch.float32)

# EnsembleDetector remains unchanged as it assumes original resolution inputs

class YoloDetectorBatch:
    def __init__(self, yolo_path):
        from ultralytics import YOLO
        import torch
        
        self.model = YOLO(yolo_path)
        
        # Set device for inference
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Move model to the appropriate device
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = 4  # Can be increased based on available memory
        
    def __call__(self, img, conf_thres=None):
        import torch
        
        
        # Run inference with proper input preparation
        with torch.no_grad():  # No gradient calculation needed for inference
            # Using stream=True but properly handling the generator
            results_gen = self.model(img, stream=True)
            # Get the first result from the generator
            results = next(results_gen)
        
        # Process results more efficiently
        boxes = results.boxes
        if len(boxes) == 0:
            return torch.zeros((0, 5))
            
        # Extract all boxes at once instead of looping
        annotations = torch.cat([
            boxes.xyxy,
            boxes.conf.unsqueeze(1)
        ], dim=1)
        
        return annotations
        
    def batch_inference(self, images):
        """Run inference on a batch of images for better performance."""
        import torch
        
        with torch.no_grad():
            results = self.model(images, half= True, stream=True, batch=len(images)) # hota 69.897, after half-> hota = 70.264
            
        all_annotations = []
        for result in results:
            if len(result.boxes) == 0:
                all_annotations.append(torch.zeros((0, 5)))
                continue
                
            boxes = result.boxes
            annotations = torch.cat([
                boxes.xyxy,
                boxes.conf.unsqueeze(1)
            ], dim=1)
            all_annotations.append(annotations)
            
        return all_annotations
    


# Faster R-CNN Detector (removed conf_threshold)
class FasterRCNNDetector:
    def __init__(self, model_path):
        anchor_sizes = tuple((int(w),) for w, _ in [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name='resnet101',
            pretrained=False,
            trainable_layers=3
        )
        self.model = FasterRCNN(
            backbone,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            rpn_pre_nms_top_n_test=4000,
            rpn_post_nms_top_n_test=400,
            rpn_nms_thresh=0.67,
            box_score_thresh=0.085,
            box_nms_thresh=0.5,
            box_detections_per_img=300
        )
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.transforms = A.Compose([
            A.Resize(height=640, width=640, always_apply=True),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __call__(self, img):
        orig_h, orig_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img_rgb)
        img_tensor = augmented['image'].to(self.device)
        img_tensor = img_tensor.unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
        
        boxes = predictions['boxes'].cpu()
        scores = predictions['scores'].cpu()
        labels = predictions['labels'].cpu()
        mask = (labels == 1)  # Only filter by class (pedestrian), no confidence threshold
        boxes = boxes[mask]
        scores = scores[mask]
        
        if len(boxes) > 0:
            scale_x = orig_w / 640
            scale_y = orig_h / 640
            boxes[:, 0] *= scale_x  # x1
            boxes[:, 1] *= scale_y  # y1
            boxes[:, 2] *= scale_x  # x2
            boxes[:, 3] *= scale_y  # y2
            annotations = torch.cat((boxes, scores.unsqueeze(1)), dim=1)
        else:
            annotations = torch.zeros((0, 5))
        
        return annotations



# In detectors.py

class EnsembleDetector(Detector):
    def __init__(self, model1: Detector, model2: Detector, model1_weight=0.7, model2_weight=0.3, iou_thresh=0.6, conf_thresh=0.3):
        self.model1 = model1
        self.model2 = model2
        self.model1_weight = model1_weight
        self.model2_weight = model2_weight
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh  # New parameter for confidence filtering

    def __call__(self, img):
        orig_h, orig_w = img.shape[:2]

        # Get predictions
        model1_preds = self.model1(img)  # Already filtered to 'person' in YoloDetector
        model2_preds = self.model2(img)

        # Prepare for WBF
        if len(model1_preds) > 0:
            yolo_boxes = model1_preds[:, :4].numpy()
            yolo_scores = model1_preds[:, 4].numpy()
            yolo_boxes_normalized = yolo_boxes / np.array([orig_w, orig_h, orig_w, orig_h])
            yolo_labels = np.zeros(len(yolo_scores))  # Class 0 for 'person'
        else:
            yolo_boxes_normalized = np.array([])
            yolo_scores = np.array([])
            yolo_labels = np.array([])

        if len(model2_preds) > 0:
            other_boxes = model2_preds[:, :4].numpy()
            other_scores = model2_preds[:, 4].numpy()
            other_boxes_normalized = other_boxes / np.array([orig_w, orig_h, orig_w, orig_h])
            other_labels = np.zeros(len(other_scores))  # Class 0 for 'person'
        else:
            other_boxes_normalized = np.array([])
            other_scores = np.array([])
            other_labels = np.array([])

        # Weighted Box Fusion
        boxes_list = [yolo_boxes_normalized, other_boxes_normalized]
        scores_list = [yolo_scores, other_scores]
        labels_list = [yolo_labels, other_labels]  # Use actual class labels
        weights = [self.model1_weight, self.model2_weight]

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=weights, iou_thr=self.iou_thresh, skip_box_thr=0.0
        )

        # Filter to only 'person' (class 0) after WBF
        person_mask = labels == 0
        boxes = boxes[person_mask]
        scores = scores[person_mask]

        # Apply confidence threshold
        conf_mask = scores > self.conf_thresh
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]

        # Scale back to original resolution
        if len(boxes) > 0:
            boxes = boxes * np.array([orig_w, orig_h, orig_w, orig_h])
            annotations = torch.tensor(np.hstack((boxes, scores[:, np.newaxis])), dtype=torch.float32)
        else:
            annotations = torch.zeros((0, 5), dtype=torch.float32)

        return annotations