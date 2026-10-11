import os
import sys
import time
import threading
import collections
import argparse
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Suppress terminal warnings & configure UTF-8 encoding
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------
# CORE INTEGRITY & AUTHOR ATTRIBUTION (HARDENED)
# -------------------------------------------------------------
__author__ = "Parshwa"
__repository__ = "https://github.com/anonymous986"
__greeting__ = "Thank you for downloading and using this software."

def display_hardened_watermark(title: str = "CAMERA-VISION-TRACKER AI"):
    border = "=" * 68
    print("\n\033[1;36m" + border + "\033[0m")
    print(f"\033[1;37m   {title.upper()}\033[0m")
    print("\033[1;36m" + "-" * 68 + "\033[0m")
    print(f"   \033[1;33mBuilt by {__author__}\033[0m")
    print(f"   \033[1;32m{__greeting__}\033[0m")
    print(f"   \033[1;34m{__repository__}\033[0m")
    print("\033[1;36m" + border + "\033[0m\n")

# Color Palette for Objects, Persons, and Specialized Categories
COLOR_PERSON = (0, 255, 128)      # Emerald Green
COLOR_FACE = (255, 100, 200)       # Violet / Magenta
COLOR_VEHICLE = (0, 165, 255)      # Amber / Orange
COLOR_ITEM = (0, 215, 255)         # Bright Cyan
COLOR_DEFAULT = (255, 255, 0)      # Electric Yellow


def compute_iou(b1, b2) -> float:
    """Computes Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def box_diou(b1, b2) -> float:
    """Computes Distance-IoU taking centroid Euclidean distance into account."""
    iou = compute_iou(b1, b2)
    c1x, c1y = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
    c2x, c2y = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
    d_center_sq = (c1x - c2x) ** 2 + (c1y - c2y) ** 2

    enc_w = max(b1[2], b2[2]) - min(b1[0], b2[0])
    enc_h = max(b1[3], b2[3]) - min(b1[1], b2[1])
    c_diag_sq = enc_w ** 2 + enc_h ** 2
    return iou - (d_center_sq / (c_diag_sq + 1e-6))


class FastMotionTracker:
    """
    Velocity-aware DIoU tracker maintaining smooth identity trajectories across frames.
    Matches across synonymous semantic classes to eliminate flickering.
    """
    def __init__(self, alpha: float = 0.75, max_lost_frames: int = 12, match_thresh: float = 0.20):
        self.alpha = alpha
        self.max_lost_frames = max_lost_frames
        self.match_thresh = match_thresh
        self.tracks = {}
        self.next_id = 0

    def update(self, detections: list) -> list:
        # Step 1: Predict positions using instantaneous velocity
        for tid, t in self.tracks.items():
            vx, vy = t["vel"]
            t["pred_box"] = np.array([
                t["box"][0] + vx,
                t["box"][1] + vy,
                t["box"][2] + vx,
                t["box"][3] + vy
            ], dtype=np.float32)

        matched_tracks = set()
        matched_dets = set()
        updated_tracks = {}

        # Step 2: Match detections with predicted track boxes
        for tid, t in self.tracks.items():
            best_score, best_idx = -1.0, -1
            for didx, (box, label, conf) in enumerate(detections):
                if didx in matched_dets:
                    continue

                l1, l2 = label.lower(), t["label"].lower()
                person_family = {"person", "person: female", "person: male", "girl: female", "boy: male", "man", "woman", "boy", "girl", "human body"}
                vehicle_family = {"car", "truck", "bus", "motorcycle", "bicycle", "van", "land vehicle", "vehicle"}
                tech_family = {"laptop", "computer", "computer keyboard", "tablet computer", "mobile phone", "cell phone", "telephone"}
                tableware_family = {"cup", "coffee cup", "mug", "tableware", "bottle", "drink"}

                same_family = (
                    (l1 == l2) or
                    (l1 in person_family and l2 in person_family) or
                    (l1 in vehicle_family and l2 in vehicle_family) or
                    (l1 in tech_family and l2 in tech_family) or
                    (l1 in tableware_family and l2 in tableware_family)
                )

                score = box_diou(t["pred_box"], box)
                # Allow high spatial overlap match even across slight label variation
                if (score > 0.45 or same_family) and score > best_score:
                    best_score = score
                    best_idx = didx

            if best_score >= self.match_thresh and best_idx != -1:
                det_box, det_label, det_conf = detections[best_idx]
                prev_center = np.array([(t["box"][0] + t["box"][2]) / 2.0, (t["box"][1] + t["box"][3]) / 2.0])
                new_center = np.array([(det_box[0] + det_box[2]) / 2.0, (det_box[1] + det_box[3]) / 2.0])
                cur_vel = new_center - prev_center

                smoothed_box = self.alpha * det_box.astype(np.float32) + (1.0 - self.alpha) * t["box"]
                smoothed_vel = 0.6 * cur_vel + 0.4 * t["vel"]

                updated_tracks[tid] = {
                    "box": smoothed_box,
                    "label": det_label,
                    "conf": 0.5 * det_conf + 0.5 * t["conf"],
                    "vel": smoothed_vel,
                    "lost": 0
                }
                matched_tracks.add(tid)
                matched_dets.add(best_idx)

        # Step 3: Motion coasting for briefly occluded objects
        for tid, t in self.tracks.items():
            if tid not in matched_tracks and t["lost"] < self.max_lost_frames:
                t["lost"] += 1
                t["box"] = t["pred_box"]
                updated_tracks[tid] = t

        # Step 4: Register new objects entering field of view
        for didx, (det_box, det_label, det_conf) in enumerate(detections):
            if didx not in matched_dets:
                updated_tracks[self.next_id] = {
                    "box": det_box.astype(np.float32),
                    "label": det_label,
                    "conf": det_conf,
                    "vel": np.array([0.0, 0.0], dtype=np.float32),
                    "lost": 0
                }
                self.next_id += 1

        self.tracks = updated_tracks
        return [
            (data["box"].astype(int), data["label"], data["conf"], tid)
            for tid, data in self.tracks.items()
            if data["lost"] <= 2
        ]


class CameraStream(threading.Thread):
    """Hardware camera capture thread with automatic multi-index negotiation."""
    def __init__(self, preferred_index: int = 0, width: int = 1280, height: int = 720, video_path: str = ""):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.running = True

        if video_path and os.path.isfile(video_path):
            print(f"[INPUT] Streaming from local video file: {video_path}")
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = None
            # Scan camera indices
            indices_to_try = [preferred_index] + [i for i in [0, 1, 2] if i != preferred_index]
            for idx in indices_to_try:
                print(f"[CAMERA] Attempting DirectShow camera index {idx}...")
                test_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(idx)
                if test_cap.isOpened():
                    ret_test, frame_test = test_cap.read()
                    if ret_test and frame_test is not None:
                        self.cap = test_cap
                        print(f"\033[1;32m[CAMERA]\033[0m Successfully locked Camera Index {idx} ({frame_test.shape[1]}x{frame_test.shape[0]})")
                        break
                    test_cap.release()

            if self.cap is None:
                print("\033[1;33m[CAMERA NOTICE]\033[0m No physical webcam responding. Creating virtual video loop.")
                self.cap = None

        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def run(self):
        while self.running:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
                else:
                    # Loop video if file ended
                    if self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
            else:
                # Synthetic animated test frame
                t = time.time()
                fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(fake_frame, "OmniVision AI - Virtual Camera Loop", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(fake_frame, f"Time: {time.strftime('%H:%M:%S')}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cx = int(640 + 200 * np.sin(t * 2))
                cy = int(360 + 100 * np.cos(t * 2))
                cv2.circle(fake_frame, (cx, cy), 50, (0, 255, 128), -1)
                cv2.rectangle(fake_frame, (300, 450), (980, 650), (100, 100, 100), -1)
                with self.lock:
                    self.ret = True
                    self.frame = fake_frame
                time.sleep(0.033)

    def get_frame(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()


class OmniVisionWorker(threading.Thread):
    """Asynchronous GPU perception engine processing YOLO detection."""
    def __init__(self, model_name_or_path: str = "yolov8x-oiv7.pt", device: str = "cuda:0", conf_threshold: float = 0.10):
        super().__init__(daemon=True)
        self.device = device
        self.conf_threshold = conf_threshold
        print(f"\033[1;36m[AI WORKER]\033[0m Initializing Universal Perception Engine on {device}...")

        # Search paths for pre-downloaded weights
        search_paths = [
            model_name_or_path,
            os.path.join(SCRIPT_DIR, model_name_or_path),
            os.path.join(r"D:\vscode\.venv", model_name_or_path),
            r"D:\vscode\.venv\yolov8x-oiv7.pt",
            "yolov8x.pt",
            "yolo11m.pt"
        ]

        resolved_model = None
        for p in search_paths:
            if os.path.isfile(p):
                resolved_model = p
                break

        if not resolved_model:
            resolved_model = "yolov8x.pt"

        print(f"\033[1;32m[AI MODEL]\033[0m Loading weights: \033[1;37m{os.path.basename(resolved_model)}\033[0m")
        self.model = YOLO(resolved_model)
        self.model.to(device)
        self.class_names = self.model.names
        print(f"[AI READY] Neural classes loaded: {len(self.class_names)} target categories.")

        self.tracker = FastMotionTracker(alpha=0.75, max_lost_frames=12, match_thresh=0.20)
        self.lock = threading.Lock()
        self.input_frame = None
        self.results = []
        self.inference_fps = 0.0
        self.running = True
        self.has_new_input = False

    def push_frame(self, frame):
        with self.lock:
            self.input_frame = frame
            self.has_new_input = True

    def get_results(self):
        with self.lock:
            return self.results, self.inference_fps

    def fuse_and_filter(self, raw_dets: list) -> list:
        """Clusters overlapping person labels while preserving fine-grained everyday objects."""
        person_like_labels = {"person", "man", "woman", "boy", "girl", "human body"}
        person_dets = []
        other_dets = []

        for box, label, conf in raw_dets:
            l_lower = label.lower().strip()
            if l_lower in person_like_labels:
                person_dets.append((box, label, conf))
            else:
                other_dets.append((box, label, conf))

        person_clusters = []
        for box, label, conf in sorted(person_dets, key=lambda x: x[2], reverse=True):
            matched = False
            for cluster in person_clusters:
                if compute_iou(box, cluster["box"]) > 0.40:
                    cluster["members"].append((box, label, conf))
                    cb = cluster["box"]
                    cluster["box"] = np.array([
                        min(cb[0], box[0]), min(cb[1], box[1]),
                        max(cb[2], box[2]), max(cb[3], box[3])
                    ], dtype=int)
                    matched = True
                    break
            if not matched:
                person_clusters.append({"box": np.array(box, dtype=int), "members": [(box, label, conf)]})

        fused_persons = []
        for cluster in person_clusters:
            member_labels = [m[1].lower() for m in cluster["members"]]
            highest_conf = max(m[2] for m in cluster["members"])
            if "woman" in member_labels:
                final_label = "Person: Female"
            elif "man" in member_labels:
                final_label = "Person: Male"
            else:
                final_label = "Person"
            fused_persons.append((cluster["box"], final_label, highest_conf))

        # Soft deduplication for same-label items
        filtered_others = []
        for o_box, o_label, o_conf in sorted(other_dets, key=lambda x: x[2], reverse=True):
            l_lower = o_label.lower().strip()
            is_dup = False
            for k_box, k_label, k_conf in filtered_others:
                if compute_iou(o_box, k_box) > 0.70 and l_lower == k_label.lower().strip():
                    is_dup = True
                    break
            if not is_dup:
                filtered_others.append((o_box, o_label, o_conf))

        return fused_persons + filtered_others

    def run(self):
        fps_hist = collections.deque(maxlen=20)
        while self.running:
            work_img = None
            with self.lock:
                if self.has_new_input and self.input_frame is not None:
                    work_img = self.input_frame
                    self.has_new_input = False

            if work_img is None:
                time.sleep(0.002)
                continue

            t_start = time.perf_counter()
            results = self.model.predict(
                work_img,
                device=self.device,
                imgsz=640,
                conf=self.conf_threshold,
                iou=0.45,
                max_det=300,
                verbose=False
            )

            raw_detections = []
            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    label = self.class_names.get(cls_id, f"Object {cls_id}")
                    raw_detections.append((xyxy, label, conf))

            fused_dets = self.fuse_and_filter(raw_detections)
            tracked_outputs = self.tracker.update(fused_dets)

            dt = time.perf_counter() - t_start
            fps_hist.append(1.0 / max(1e-4, dt))

            with self.lock:
                self.results = tracked_outputs
                self.inference_fps = float(np.mean(fps_hist))


def main():
    parser = argparse.ArgumentParser(description="OmniVision Camera AI: Universal 'All Things Detection' Perception Engine")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--video", default="", help="Path to video file instead of webcam")
    parser.add_argument("--conf", type=float, default=0.10, help="Detection confidence threshold (default: 0.10)")
    parser.add_argument("--model", default="yolov8x-oiv7.pt", help="Model weights (default: yolov8x-oiv7.pt)")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    display_hardened_watermark("CAMERA-VISION-TRACKER - REAL-TIME PERCEPTION CORE")

    # Start camera capture
    cam = CameraStream(preferred_index=args.cam, width=1280, height=720, video_path=args.video)
    cam.start()

    # Start neural perception worker
    worker = OmniVisionWorker(model_name_or_path=args.model, device=device, conf_threshold=args.conf)
    worker.start()

    window_title = "Camera Vision Tracker - Built by Parshwa"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    print(f"\n[ACTIVE] Live Camera Perception Active. Hotkeys: [q] Quit | [c] Clear Tracks | [s] Screenshot")

    try:
        while True:
            ret, frame = cam.get_frame()
            if not ret or frame is None:
                time.sleep(0.002)
                continue

            worker.push_frame(frame)
            detections, ai_fps = worker.get_results()

            display_frame = frame.copy()

            # Render Bounding Boxes & Badges
            for xyxy, label, conf, track_id in detections:
                x1, y1, x2, y2 = xyxy
                l_lower = label.lower()

                if "person" in l_lower or "girl" in l_lower or "boy" in l_lower or "man" in l_lower or "woman" in l_lower:
                    box_color = COLOR_PERSON
                elif "face" in l_lower or "head" in l_lower:
                    box_color = COLOR_FACE
                elif any(v in l_lower for v in ["car", "vehicle", "truck", "motorcycle", "bicycle"]):
                    box_color = COLOR_VEHICLE
                else:
                    box_color = COLOR_ITEM

                # Clean Bounding Box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

                # Label Pill
                tag = f"#{track_id} {label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(display_frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), box_color, -1)
                cv2.putText(display_frame, tag, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            # Top Dashboard Overlay
            person_count = sum(1 for _, l, _, _ in detections if any(p in l.lower() for p in ["person", "man", "woman", "boy", "girl"]))
            total_objects = len(detections)
            cv2.rectangle(display_frame, (10, 10), (450, 80), (20, 20, 20), -1)
            cv2.rectangle(display_frame, (10, 10), (450, 80), (60, 60, 60), 1)

            cv2.putText(display_frame, f"Vision Core  |  by {__author__}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {ai_fps:.1f}  |  Objects: {total_objects}  |  Persons: {person_count}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Permanent Bottom Watermark Banner
            h_frame, w_frame = display_frame.shape[:2]
            cv2.rectangle(display_frame, (0, h_frame - 24), (w_frame, h_frame), (14, 14, 14), -1)
            cv2.putText(
                display_frame,
                f"Built by {__author__}  |  {__repository__}",
                (12, h_frame - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 230, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(window_title, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                shot_path = f"omnivision_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(shot_path, display_frame)
                print(f"[SNAPSHOT] Saved: {shot_path}")
            elif key == ord('c'):
                worker.tracker.tracks.clear()
                print("[TRACKER] Track history cleared.")

    finally:
        worker.running = False
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
