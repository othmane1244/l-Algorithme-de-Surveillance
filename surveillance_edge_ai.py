# Surveillance Edge AI - Détection et suivi d'objets en temps réel

# bibliothèques nécessaires
import argparse 
import time
import cv2  
import numpy as np
from ultralytics import YOLO

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
    # affichage
    "window_name": "Surveillance Edge AI",
    "bbox_color": (0, 255, 0), # couleur des boîtes englobantes
    "id_color": (255, 0, 0), # couleur des IDs de suivi
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.6,
    "font_thickness": 2,
    "fps_color": (0, 255, 255),# couleur du texte du FPS
    "fps_interval": 10, # nombre de frames pour calculer le FPS
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

def draw_detections(frame: np.ndarray, results,fps: float, config: dict) -> np.ndarray:
    """Dessine les détections et les IDs de suivi sur la frame."""
    person_count = 0

    for result in results:
        boxes = result.boxes # Accès aux boîtes englobantes (bounding boxes) de la détection
        
        if boxes is  None or len(boxes) == 0: # Si aucune boîte n'est détectée, passer à la frame suivante
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val =float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            person_count += 1
            # dessiner la boîte englobante
            cv2.rectangle(frame, (x1, y1), (x2, y2), config["bbox_color"], config["font_thickness"])
            # dessiner l'ID de suivi
            label = f"ID: {track_id} Conf: {conf_val:.2f}" if track_id != -1 else "ID: N/A"
            (tw, th), bl = cv2.getTextSize(label, config["font"], config["font_scale"], config["font_thickness"])
            y_top = max(y1 - th - bl - 5, 0)
            cv2.putText(frame, label, (x1 + 3, y_top), config["font"], config["font_scale"], config["id_color"], config["font_thickness"])
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (200, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.2f}", (15, 45), config["font"], config["font_scale"], config["fps_color"], config["font_thickness"], 2,  cv2.LINE_AA)
    cv2.putText(frame, f"Personnes : {person_count}", (130, 42), config["font"], config["font_scale"], config["fps_color"], config["font_thickness"], 2, cv2.LINE_AA)

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
    fps = 0.0 
    frame_count = 0
    t_start = time.perf_counter()
    N = config["fps_interval"]

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

        # -- 7. Gestion clavier (1 ms, non bloquant) --------------------------
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Arret utilisateur (touche 'q').")
            break

    # -- Liberation propre des ressources ------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Ressources liberees. Programme termine.")

def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Surveillance Edge AI - Détection et suivi d'objets en temps réel")
    parser.add_argument("--source", type=str, default=CONFIG["source"], help="Source vidéo (0 pour webcam ou chemin vers fichier vidéo)")
    parser.add_argument("--model", type=str, default=CONFIG["model_path"], help="Chemin vers le modèle YOLO")
    parser.add_argument("--tracker", type=str, default=CONFIG["tracker"], help="Configuration du tracker (ex: 'bytetrack.yaml')")
    parser.add_argument("--conf", type=float, default=CONFIG["confidence"], help="Seuil de confiance pour les détections")
    parser.add_argument("--classes", type=int, nargs="+", default=CONFIG["classes"], help="Classes à détecter (ex: 0 pour personne)")
    return parser.parse_args()

if __name__ == "__main__":    
    args = parse_args()
    CONFIG["source"] = int(args.source) if str(args.source).isdigit() else args.source
    CONFIG["model_path"] = args.model
    CONFIG["confidence"] = args.conf

    run_surveillance(CONFIG)
