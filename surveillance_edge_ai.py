# Surveillance Edge AI - Détection et suivi d'objets en temps réel

# bibliothèques nécessaires
import argparse 
import time
import cv2  
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# configuration 
CONFIG = {
    # source de video (0 pour la webcam, ou chemin vers un fichier vidéo)
    "source": 0,
    # prepocessing des frames
    "frame_size": (640, 640),
    # inference 
    "model_path": "yolov8n.pt",
    "classes": [0], # classes à détecter (0 = personne)
    "confidence": 0.5,
    # tracking
    "tracker": "bytetrack.yaml", 
    #logique metier 
    "line_pt1": (100, 200), # point de départ de la ligne virtuelle
    "line_pt2": (500, 200), # point d'arrivée de la ligne
    "alert_color": (0, 0, 255), # couleur de l'alerte
    "line_color": (255, 255, 0), # couleur de la ligne virtuelle
    "line_thickness": 2, # épaisseur de la ligne virtuelle
    # affichage
    "window_name": "Surveillance Edge AI",
    "bbox_color": (0, 255, 0), # couleur des boîtes englobantes
    "id_color": (255, 0, 0), # couleur des IDs de suivi
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.6,
    "font_thickness": 2,
    "fps_color": (0, 255, 255),# couleur du texte du FPS
    "fps_interval": 10, # nombre de frames pour calculer le FPS
    "history_length": 2, # nombre de frames pour le suivi historique (ex: pour les trajectoires)
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
    """Calcule le déterminant pour déterminer de quel côté de la ligne se trouve le point."""
    x, y = point
    x_a, y_a = line_pt1
    x_b, y_b = line_pt2

    d = (x -x_a) * (y_b - y_a) - (y - y_a) * (x_b - x_a)
    return d

def get_centroid(box):
    """Calcule le centroïde d'une boîte englobante."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def check_line_crossing(track_id, current_centroid, track_history, line_pt1, line_pt2):
    """Vérifie si un objet a traversé la ligne virtuelle."""
    if track_id not in track_history or len(track_history[track_id]) < 2:
        return False, None
    
    prev_centroid = track_history[track_id][-2]  
    d_prev = calculate_determinant(prev_centroid, line_pt1, line_pt2)
    d_curr = calculate_determinant(current_centroid, line_pt1, line_pt2)

    if d_prev * d_curr < 0:  
        if d_prev > 0 and d_curr < 0:
            direction = "left_to_right"
        else:
            direction = "right_to_left"
        return True, direction
    
    return False, None

def draw_detections(frame: np.ndarray, results, fps: float, config: dict, 
                   track_history: dict, crossed_ids: set) -> np.ndarray:
    """Dessine les détections, les IDs de suivi, la ligne virtuelle et les alertes."""
    person_count = 0
    h, w = frame.shape[:2]
    
    # Dessiner la ligne virtuelle
    cv2.line(frame, config["line_pt1"], config["line_pt2"], 
             config["line_color"], config["line_thickness"])
    
    # Ajouter un label pour la ligne
    mid_x = (config["line_pt1"][0] + config["line_pt2"][0]) // 2
    mid_y = (config["line_pt1"][1] + config["line_pt2"][1]) // 2
    cv2.putText(frame, "LIGNE DE SECURITE", (mid_x - 80, mid_y), 
                config["font"], 0.5, config["line_color"], 1)

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
            
            # Calculer le centroïde
            centroid = get_centroid((x1, y1, x2, y2))
            
            # Mettre à jour l'historique des positions
            track_history[track_id].append(centroid)
            # Garder seulement les N dernières positions
            if len(track_history[track_id]) > config["history_length"]:
                track_history[track_id].pop(0)
            
            # Vérifier le franchissement de ligne
            crossed, direction = check_line_crossing(
                track_id, centroid, track_history, 
                config["line_pt1"], config["line_pt2"]
            )
            
            if crossed:
                crossed_ids.add(track_id)
                print(f"[ALERTE] ID {track_id} a franchi la ligne! Direction: {direction}")
            
            # Déterminer la couleur de la boîte (rouge si alerte, vert sinon)
            box_color = config["alert_color"] if track_id in crossed_ids else config["bbox_color"]
            
            # Dessiner la boîte englobante
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, config["font_thickness"])
            
            # Dessiner le centroïde
            cv2.circle(frame, centroid, 4, (0, 0, 255), -1)
            
            # Dessiner la trajectoire (ligne entre les positions précédentes)
            if len(track_history[track_id]) > 1:
                points = np.array(track_history[track_id], np.int32)
                cv2.polylines(frame, [points], False, (255, 255, 255), 1)
            
            # Dessiner l'ID de suivi et la confiance
            alert_text = " [ALERTE]" if track_id in crossed_ids else ""
            label = f"ID: {track_id} Conf: {conf_val:.2f}{alert_text}"
            
            (tw, th), bl = cv2.getTextSize(label, config["font"], 
                                           config["font_scale"], config["font_thickness"])
            y_top = max(y1 - th - bl - 5, 0)
            
            # Fond pour le texte
            cv2.rectangle(frame, (x1, y_top), (x1 + tw, y_top + th + bl), 
                         (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, y_top + th), config["font"], 
                       config["font_scale"], box_color, config["font_thickness"])

    # HUD (Heads-Up Display) - Informations en overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (15, 35), config["font"], 
               config["font_scale"], config["fps_color"], config["font_thickness"], 
               cv2.LINE_AA)
    
    # Compteur de personnes
    cv2.putText(frame, f"Personnes: {person_count}", (15, 60), config["font"], 
               config["font_scale"], config["fps_color"], config["font_thickness"], 
               cv2.LINE_AA)
    
    # Compteur d'alertes
    cv2.putText(frame, f"Alertes: {len(crossed_ids)}", (15, 85), config["font"], 
               config["font_scale"], config["alert_color"], config["font_thickness"], 
               cv2.LINE_AA)

    return frame

def init_capture(source) -> cv2.VideoCapture:
    """Initialise la capture vidéo."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la source vidéo: {source}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # === Raspberry Pi 5 -- options supplementaires (decommenter si besoin) ===
    # Codec MJPEG pour USB cameras a haute frequence :
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # Backend V4L2 :
    # cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    # =========================================================================

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    print(f"Capture vidéo initialisée: {w}x{h}")

    return cap

def run_surveillance(config: dict) -> None:
    
    model = YOLO(config["model_path"])
    cap = init_capture(config["source"])
    # Initialisation de l'historique des trajectoires
    track_history = defaultdict(list)
    crossed_ids = set()  # Ensemble des IDs ayant franchi la ligne
    fps = 0.0 
    frame_count = 0
    t_start = time.perf_counter()
    N = config["fps_interval"]
    print("[INFO] Démarrage de la surveillance avec détection de franchissement...")
    print(f"[INFO] Ligne définie de {config['line_pt1']} à {config['line_pt2']}")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AVERT] Fin du flux ou frame corrompue. Arret.")
            break

        frame_norm = preprocess_frame(frame, config["frame_size"])
        results = run_inference_and_tracking(model, frame, config)


        frame_count += 1
        if frame_count % N == 0:
            t_now = time.perf_counter()
            elapsed = t_now - t_start
            fps = N / (elapsed + 1e-9)   # Evite division par zero
            t_start = t_now

        frame_display = draw_detections(frame, results, fps, config)
        cv2.imshow(config["window_name"], frame_display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Arret utilisateur (touche 'q').")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Ressources liberees. Programme termine.")
    print(f"[INFO] Total d'alertes déclenchées: {len(crossed_ids)}")

def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Surveillance Edge AI - Détection et suivi d'objets en temps réel")
    parser.add_argument("--source", type=str, default=CONFIG["source"], help="Source vidéo (0 pour webcam ou chemin vers fichier vidéo)")
    parser.add_argument("--model", type=str, default=CONFIG["model_path"], help="Chemin vers le modèle YOLO")
    parser.add_argument("--tracker", type=str, default=CONFIG["tracker"], help="Configuration du tracker (ex: 'bytetrack.yaml')")
    parser.add_argument("--conf", type=float, default=CONFIG["confidence"], help="Seuil de confiance pour les détections")
    parser.add_argument("--classes", type=int, nargs="+", default=CONFIG["classes"], help="Classes à détecter (ex: 0 pour personne)")
    parser.add_argument("--line-x1", type=int, default=CONFIG["line_pt1"][0], 
                       help="Coordonnée X du point 1 de la ligne")
    parser.add_argument("--line-y1", type=int, default=CONFIG["line_pt1"][1], 
                       help="Coordonnée Y du point 1 de la ligne")
    parser.add_argument("--line-x2", type=int, default=CONFIG["line_pt2"][0], 
                       help="Coordonnée X du point 2 de la ligne")
    parser.add_argument("--line-y2", type=int, default=CONFIG["line_pt2"][1], 
                       help="Coordonnée Y du point 2 de la ligne")
    return parser.parse_args()

if __name__ == "__main__":    
    args = parse_args()
    CONFIG["source"] = int(args.source) if str(args.source).isdigit() else args.source
    CONFIG["model_path"] = args.model
    CONFIG["confidence"] = args.conf
    CONFIG["line_pt1"] = (args.line_x1, args.line_y1)
    CONFIG["line_pt2"] = (args.line_x2, args.line_y2)

    run_surveillance(CONFIG)
