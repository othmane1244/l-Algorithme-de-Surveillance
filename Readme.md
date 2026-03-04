# Edge AI Video Surveillance

Détection et suivi de personnes en temps réel avec **YOLOv8n** + **ByteTrack**, conçu pour tourner sur PC et **Raspberry Pi 5**.

---

## Aperçu

```
Flux vidéo (camera / fichier / RTSP)
        │
        ▼
[A] Preprocessing       ──  resize 640×640 · BGR→RGB · normalisation [0,1]
        │
        ▼
[B] Inférence YOLOv8n   ──  détection personnes (classe COCO 0) · conf ≥ 0.5
        │
        ▼
[C] Tracking ByteTrack  ──  IDs uniques persistants entre frames
        │
        ▼
Affichage annoté         ──  BBox · ID · FPS · compteur personnes
```

---

## Prérequis

- Python **3.9+**
- Webcam USB, fichier vidéo ou flux RTSP
- (Optionnel) GPU CUDA pour de meilleures performances sur PC

---

## Installation

```bash
# 1. Cloner le projet
git clone https://github.com/votre-repo/edge-ai-surveillance.git
cd edge-ai-surveillance

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
#ou
pip install opencv-python ultralytics numpy
```

> Le modèle `yolov8n.pt` est téléchargé automatiquement au premier lancement.

---

## Utilisation

```bash
# Webcam par défaut (index 0)
python surveillance_edge_ai.py

# Caméra USB spécifique
python surveillance_edge_ai.py --source 0

# Fichier vidéo local
python surveillance_edge_ai.py --source video.mp4

# Flux RTSP (IP camera)
python surveillance_edge_ai.py --source rtsp://192.168.1.10/stream1

# Paramètres personnalisés
python surveillance_edge_ai.py --source 0 --conf 0.4 --model yolov8s.pt
```

### Arguments disponibles

| Argument | Défaut | Description |
|---|---|---|
| `--source` | `0` | Source vidéo : index caméra, chemin fichier ou URL RTSP |
| `--model` | `yolov8n.pt` | Modèle YOLOv8 à utiliser |
| `--conf` | `0.5` | Seuil de confiance minimal (0.0 – 1.0) |

**Quitter :** appuyer sur `q` dans la fenêtre d'affichage.

---

## Structure du projet

```
edge-ai-surveillance/
├── surveillance_edge_ai.py   # Script principal
├── requirements.txt          # Dépendances Python
└── README.md                 # Ce fichier
```

---

## Architecture du code

| Fonction | Étape | Rôle |
|---|---|---|
| `preprocess_frame()` | A | Resize · BGR→RGB · normalisation float32 |
| `run_inference_and_tracking()` | B + C | Inférence YOLOv8n + tracking ByteTrack |
| `draw_detections()` | Rendu | BBox · label ID · FPS · compteur |
| `init_capture()` | Init | Ouverture et configuration de la source vidéo |
| `run_surveillance()` | Main | Boucle principale optimisée FPS |

La configuration globale `CONFIG` centralise tous les paramètres (seuils, couleurs, modèle, tracker) pour faciliter le portage et la maintenance.

---

## Portage Raspberry Pi 5

Le script est directement compatible. Pour de meilleures performances :

**1. Activer le codec MJPEG (caméra USB)**

Dans `init_capture()`, décommenter :
```python
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 30)
```

**2. Activer le backend V4L2**

```python
cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
```

**3. Exporter le modèle en ONNX ou TFLite** pour une inférence accélérée :

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx")    # -> yolov8n.onnx
# ou
model.export(format="tflite") # -> yolov8n_float32.tflite
```

Puis modifier `CONFIG["model_path"]` avec le chemin du modèle exporté.

---

## Dépendances

| Paquet | Version min. | Rôle |
|---|---|---|
| `opencv-python` | 4.8.0 | Capture vidéo & rendu |
| `ultralytics` | 8.0.0 | YOLOv8 + ByteTrack intégré |
| `numpy` | 1.24.0 | Calcul matriciel |

---

## Performances indicatives

| Plateforme | Modèle | FPS (approx.) |
|---|---|---|
| PC — CPU (i7) | yolov8n.pt | ~25–35 FPS |
| PC — GPU (RTX 3060) | yolov8n.pt | ~80–120 FPS |
| Raspberry Pi 5 | yolov8n.pt (CPU) | ~5–10 FPS |
| Raspberry Pi 5 | yolov8n.onnx | ~10–18 FPS |

> Les FPS varient selon la résolution source, le nombre de personnes détectées et la charge système.

---

## Licence

Ce projet est distribué sous licence MIT. Voir `LICENSE` pour plus de détails.