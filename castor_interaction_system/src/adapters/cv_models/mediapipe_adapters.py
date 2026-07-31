import cv2
from typing import Any
from ultralytics import YOLO
from src.core.entities import Keypoint, PoseData
from src.core.interfaces import IPoseEstimator

class MediaPipePoseAdapter(IPoseEstimator):
    def __init__(self, hardware_config: Any):
        self.model = YOLO("yolov8n-pose.pt")
        self.device = getattr(hardware_config, "device", "cpu")
        self.model.to(self.device)

    def estimate_pose(self, frame: Any, person_bbox: Any) -> PoseData:
        h, w, _ = frame.shape
        
        if hasattr(person_bbox, 'xmin') and hasattr(person_bbox, 'ymin'):
            xmin, ymin, xmax, ymax = person_bbox.xmin, person_bbox.ymin, person_bbox.xmax, person_bbox.ymax
        elif hasattr(person_bbox, 'x1'):
            xmin, ymin, xmax, ymax = person_bbox.x1, person_bbox.y1, person_bbox.x2, person_bbox.y2
        else:
            try:
                xmin, ymin, xmax, ymax = map(float, person_bbox)
            except Exception:
                return self._empty_pose()

        xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
        xmax, ymax = min(w, int(xmax)), min(h, int(ymax))
        
        if xmin >= xmax or ymin >= ymax:
            return self._empty_pose()

        cropped_person = frame[ymin:ymax, xmin:xmax]
        if cropped_person.size == 0:
            return self._empty_pose()

        results = self.model(cropped_person, verbose=False, device=self.device)

        keypoints_list = []
        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kpts_xy = results[0].keypoints.xy[0].cpu().numpy()
            kpts_conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else [1.0] * len(kpts_xy)

            for pt, conf in zip(kpts_xy, kpts_conf):
                abs_x = float(pt[0] + xmin)
                abs_y = float(pt[1] + ymin)
                
                try:
                    kpt = Keypoint(x=abs_x, y=abs_y, confidence=float(conf))
                except TypeError:
                    try:
                        kpt = Keypoint(x=abs_x, y=abs_y, score=float(conf))
                    except TypeError:
                        try:
                            kpt = Keypoint(abs_x, abs_y, float(conf))
                        except Exception:
                            continue
                
                keypoints_list.append(kpt)

        return self._build_pose_data(keypoints_list)

    def _empty_pose(self) -> PoseData:
        return self._build_pose_data([])

    def _build_pose_data(self, keypoints_list) -> PoseData:
        dummy_vector = [0.0, 0.0, 0.0]
        
        # Mapeia a lista de keypoints do COCO para um dicionário compatível com .get()
        coco_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        landmarks_dict = {}
        for idx, kpt in enumerate(keypoints_list):
            if idx < len(coco_names):
                landmarks_dict[coco_names[idx]] = kpt
            landmarks_dict[idx] = kpt

        try:
            return PoseData(
                landmarks=landmarks_dict,
                body_orientation_vector=dummy_vector,
                head_orientation_vector=dummy_vector
            )
        except TypeError:
            try:
                return PoseData(
                    landmarks_dict,
                    dummy_vector,
                    dummy_vector
                )
            except Exception:
                p = PoseData.__new__(PoseData)
                p.landmarks = landmarks_dict
                p.body_orientation_vector = dummy_vector
                p.head_orientation_vector = dummy_vector
                return p

    def release(self) -> None:
        pass