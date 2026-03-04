# Surveillance Edge AI - Détection et suivi d'objets en temps réel

import argparse 
import time
import cv2  
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

CONFIG = {
    "source": 0,
    "frame_size": (640, 640),
    "model_path": "yolov8n.pt",
    "classes": [0],
    "confidence": 0.5,
    "tracker": "bytetrack.yaml", 
    #"line_pt1": (100, 200),
    #"line_pt2": (500, 200),
    # VERTICAL (nouveau) — ligne au centre de l'image 1280x720
    "line_pt1": (640, 0),    # x=640 (milieu), y=0 (haut)
    "line_pt2": (640, 720),  # x=640 (milieu), y=720 (bas)
    "alert_color": (0, 0, 255),
    "line_color": (255, 255, 0),
    "line_thickness": 2,
    "window_name": "Surveillance Edge AI",
    "bbox_color": (0, 255, 0),
    "id_color": (255, 0, 0),
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.6,
    "font_thickness": 2,
    "fps_color": (0, 255, 255),
    "fps_interval": 10,
    "history_length": 2,
}

def preprocess_frame(frame: np.ndarray, target_size: tuple) -> np.ndarray:
    """Redimensionne et normalise la frame pour l'inference."""
    frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    norm = rgb_frame.astype(np.float32) / 255.0
    return norm

def run_inference_and_tracking(model: YOLO, frame: np.ndarray, config: dict):
    """Exécute l'inference et le tracking sur la frame donnée."""
    results = model.track(
        source=frame,
        persist=True,
        tracker=config["tracker"],
        conf=config["confidence"],
        classes=config["classes"],
        imgsz=config["frame_size"][0],
        verbose=False,
    )
    return results

def calculate_determinant(point, line_pt1, line_pt2):
    """
    Calcule le déterminant pour déterminer de quel côté de la ligne se trouve le point.
    d = (x - x_A)(y_B - y_A) - (y - y_A)(x_B - x_A)
    """
    x, y = point
    x_a, y_a = line_pt1
    x_b, y_b = line_pt2
    return (x - x_a) * (y_b - y_a) - (y - y_a) * (x_b - x_a)

def get_centroid(box):
    """Calcule le centroïde d'une boîte englobante."""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def check_line_crossing(track_id, current_centroid, track_history, line_pt1, line_pt2):
    """
    Vérifie si un objet a traversé la ligne virtuelle.

    DOIT être appelé AVANT update_track_history() :
      - history[-1] = position t-1 (la dernière enregistrée)
      - current_centroid = position t (pas encore enregistrée)
    Ainsi d_prev et d_curr comparent bien deux frames consécutives distinctes.
    """
    if track_id not in track_history or len(track_history[track_id]) < 1:
        return False, None

    prev_centroid = track_history[track_id][-1]   # t-1 : dernière pos enregistrée

    d_prev = calculate_determinant(prev_centroid,    line_pt1, line_pt2)
    d_curr = calculate_determinant(current_centroid, line_pt1, line_pt2)

    if d_prev * d_curr < 0:
        direction = "left_to_right" if d_prev > 0 else "right_to_left"
        return True, direction

    return False, None

def update_track_history(track_id, centroid, track_history, history_length):
    """
    Ajoute le centroïde courant dans l'historique APRÈS le check.
    Limite la taille à history_length positions.
    """
    track_history[track_id].append(centroid)
    if len(track_history[track_id]) > history_length:
        track_history[track_id].pop(0)

def draw_detections(frame: np.ndarray, results, fps: float, config: dict,
                    track_history: dict, crossed_ids: set) -> np.ndarray:
    """Dessine les détections, IDs, ligne virtuelle et alertes."""
    person_count = 0

    # -- Ligne virtuelle ------------------------------------------------------
    cv2.line(frame, config["line_pt1"], config["line_pt2"],
             config["line_color"], config["line_thickness"])
    mid_x = (config["line_pt1"][0] + config["line_pt2"][0]) // 2
    mid_y = (config["line_pt1"][1] + config["line_pt2"][1]) // 2
    cv2.putText(frame, "LIGNE DE SECURITE", (mid_x - 80, mid_y),
                config["font"], 0.5, config["line_color"], 1, cv2.LINE_AA)

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            if track_id == -1:
                continue

            person_count += 1
            centroid = get_centroid((x1, y1, x2, y2))

            # ── ORDRE CRITIQUE : check t-1 vs t, PUIS enregistre t ───────────
            crossed, direction = check_line_crossing(
                track_id, centroid, track_history,
                config["line_pt1"], config["line_pt2"]
            )
            update_track_history(
                track_id, centroid, track_history, config["history_length"]
            )
            # ─────────────────────────────────────────────────────────────────

            if crossed:
                crossed_ids.add(track_id)
                print(f"[ALERTE] ID {track_id} a franchi la ligne! Direction: {direction}")

            box_color = config["alert_color"] if track_id in crossed_ids else config["bbox_color"]

            # -- Boîte englobante ---------------------------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, config["font_thickness"])

            # -- Centroïde ----------------------------------------------------
            cv2.circle(frame, centroid, 4, (0, 0, 255), -1, cv2.LINE_AA)

            # -- Trajectoire --------------------------------------------------
            if len(track_history[track_id]) > 1:
                points = np.array(track_history[track_id], np.int32)
                cv2.polylines(frame, [points], False, (255, 255, 255), 1, cv2.LINE_AA)

            # -- Label ID + confiance -----------------------------------------
            alert_text = " [ALERTE]" if track_id in crossed_ids else ""
            label = f"ID:{track_id} {conf_val:.2f}{alert_text}"
            (tw, th), bl = cv2.getTextSize(label, config["font"],
                                           config["font_scale"], config["font_thickness"])
            y_top = max(y1 - th - bl - 5, 0)
            cv2.rectangle(frame, (x1, y_top), (x1 + tw + 4, y_top + th + bl),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (x1 + 2, y_top + th),
                        config["font"], config["font_scale"],
                        box_color, config["font_thickness"], cv2.LINE_AA)

    # -- HUD ------------------------------------------------------------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.2f}", (15, 35),
                config["font"], config["font_scale"],
                config["fps_color"], config["font_thickness"], cv2.LINE_AA)
    cv2.putText(frame, f"Personnes: {person_count}", (15, 60),
                config["font"], config["font_scale"],
                config["fps_color"], config["font_thickness"], cv2.LINE_AA)
    cv2.putText(frame, f"Alertes: {len(crossed_ids)}", (15, 85),
                config["font"], config["font_scale"],
                config["alert_color"], config["font_thickness"], cv2.LINE_AA)

    return frame

def init_capture(source) -> cv2.VideoCapture:
    """Initialise la capture vidéo."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la source vidéo: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # === Raspberry Pi 5 (décommenter si besoin) ===
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Capture vidéo initialisée: {w}x{h}")
    return cap

def run_surveillance(config: dict) -> None:
    model = YOLO(config["model_path"])
    cap = init_capture(config["source"])

    track_history = defaultdict(list)
    crossed_ids   = set()
    fps           = 0.0
    frame_count   = 0
    t_start       = time.perf_counter()
    N             = config["fps_interval"]

    print("[INFO] Démarrage de la surveillance avec détection de franchissement...")
    print(f"[INFO] Ligne définie de {config['line_pt1']} à {config['line_pt2']}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AVERT] Fin du flux ou frame corrompue. Arret.")
            break

        _ = preprocess_frame(frame, config["frame_size"])
        results = run_inference_and_tracking(model, frame, config)

        frame_count += 1
        if frame_count % N == 0:
            t_now   = time.perf_counter()
            fps     = N / (t_now - t_start + 1e-9)
            t_start = t_now

        frame_display = draw_detections(frame, results, fps, config,
                                        track_history, crossed_ids)
        cv2.imshow(config["window_name"], frame_display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Arret utilisateur (touche 'q').")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Ressources liberees. Programme termine.")
    print(f"[INFO] Total d'alertes déclenchées: {len(crossed_ids)}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Surveillance Edge AI - Détection et suivi avec Line Crossing"
    )
    parser.add_argument("--source",  type=str,   default=CONFIG["source"])
    parser.add_argument("--model",   type=str,   default=CONFIG["model_path"])
    parser.add_argument("--tracker", type=str,   default=CONFIG["tracker"])
    parser.add_argument("--conf",    type=float, default=CONFIG["confidence"])
    parser.add_argument("--classes", type=int, nargs="+", default=CONFIG["classes"])
    parser.add_argument("--line-x1", type=int,   default=CONFIG["line_pt1"][0])
    parser.add_argument("--line-y1", type=int,   default=CONFIG["line_pt1"][1])
    parser.add_argument("--line-x2", type=int,   default=CONFIG["line_pt2"][0])
    parser.add_argument("--line-y2", type=int,   default=CONFIG["line_pt2"][1])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    CONFIG["source"]     = int(args.source) if str(args.source).isdigit() else args.source
    CONFIG["model_path"] = args.model
    CONFIG["confidence"] = args.conf
    CONFIG["line_pt1"]   = (args.line_x1, args.line_y1)
    CONFIG["line_pt2"]   = (args.line_x2, args.line_y2)
    run_surveillance(CONFIG)