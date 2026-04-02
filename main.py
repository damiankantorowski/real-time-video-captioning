from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import torch
import numpy as np
from transformers import VivitImageProcessor, VivitForVideoClassification
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoTokenizer, AutoModel
import cv2
import base64
from collections import deque
import threading
import time
import os
import json
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Stałe dla InternVL
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """Buduje transformację obrazu dla InternVL"""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Znajduje najbliższy aspect ratio dla InternVL"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamiczne przetwarzanie obrazu dla InternVL"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def preprocess_image_for_internvl(image, input_size=448, max_num=6):
    """Przetwarza obraz PIL dla InternVL"""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Konfiguracja modelu ViViT
MODEL_NAME = "google/vivit-b-16x2-kinetics400"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Ładowanie modelu ViViT na urządzeniu: {device}")
processor = VivitImageProcessor.from_pretrained(MODEL_NAME)
model = VivitForVideoClassification.from_pretrained(
    MODEL_NAME,
    ignore_mismatched_sizes=False,
)

# Załaduj etykiety Kinetics-400
print("[*] Ladowanie etykiet Kinetics-400 z pliku lokalnego...")
try:
    with open("kinetics400_labels.json", "r") as f:
        labels_data = json.load(f)
        id2label = {int(k): v for k, v in labels_data.items()}
        model.config.id2label = id2label
        model.config.label2id = {v: k for k, v in id2label.items()}
        print(f"[OK] Zaladowano {len(id2label)} etykiet klas")
except Exception as e:
    print(f"[!] Blad ladowania etykiet: {e}")
    model.config.id2label = {i: f"Activity {i}" for i in range(400)}

model = model.to(device)
model.eval()
print("Model ViViT załadowany!")
sample_label = model.config.id2label.get(0, "Unknown")
print(f"Przykładowa klasa: {sample_label}")

# Konfiguracja modelu RT-DETR (detekcja obiektów)
print(f"\n[*] Ladowanie modelu RT-DETR na urzadzeniu: {device}")
try:
    DETECTION_MODEL_NAME = "PekingU/rtdetr_r50vd"
    detection_processor = RTDetrImageProcessor.from_pretrained(DETECTION_MODEL_NAME)
    detection_model = RTDetrForObjectDetection.from_pretrained(DETECTION_MODEL_NAME)
    detection_model = detection_model.to(device)
    detection_model.eval()
    print("[OK] Model RT-DETR zaladowany!")
except Exception as e:
    print(f"[!] Blad ladowania modelu detekcji: {e}")
    print("[*] Detekcja obiektow wylaczona")
    detection_model = None
    detection_processor = None

# Konfiguracja modelu ViT (klasyfikacja pojedynczych obrazów)
print(f"\n[*] Ladowanie modelu ViT na urzadzeniu: {device}")
try:
    VIT_MODEL_NAME = "google/vit-base-patch16-224"
    vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
    vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME)
    vit_model = vit_model.to(device)
    vit_model.eval()
    print(f"[OK] Model ViT zaladowany! Klasy: {len(vit_model.config.id2label)}")
except Exception as e:
    print(f"[!] Blad ladowania modelu ViT: {e}")
    print("[*] Klasyfikacja obrazow wylaczona")
    vit_model = None
    vit_processor = None

# Konfiguracja modelu InternVL3_5-4B (Vision-Language Model)
print(f"\n[*] Ladowanie modelu InternVL3_5-4B na urzadzeniu: {device}")
try:
    INTERNVL_MODEL_NAME = "OpenGVLab/InternVL3_5-4B-Flash"
    internvl_tokenizer = AutoTokenizer.from_pretrained(INTERNVL_MODEL_NAME, trust_remote_code=True)
    internvl_model = AutoModel.from_pretrained(
        INTERNVL_MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()
    if device == "cuda":
        internvl_model = internvl_model.to(device)
    print(f"[OK] Model InternVL3_5-4B-Flash zaladowany!")
except Exception as e:
    print(f"[!] Blad ladowania modelu InternVL: {e}")
    print("[*] Generowanie opisow wideo wylaczone")
    internvl_model = None
    internvl_tokenizer = None

# Bufor dla ramek wideo (ViViT wymaga sekwencji ramek)
FRAME_BUFFER_SIZE = 32  # ViViT standardowo używa 32 ramek
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
buffer_lock = threading.Lock()

# Cache dla ostatniej predykcji
last_prediction = {
    "top_classes": [
        {"class": "Oczekiwanie...", "confidence": 0.0}
    ],
    "objects": [],
    "vit_classes": [],
    "video_description": "Oczekiwanie..."
}
prediction_lock = threading.Lock()

# Cache dla ostatniej ramki do detekcji
last_frame = None
last_frame_lock = threading.Lock()


def process_frame(frame_base64):
    """Dekoduje ramkę z base64 i dodaje do bufora"""
    global last_frame
    try:
        # Dekoduj base64
        img_data = base64.b64decode(frame_base64.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Konwersja BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Zapisz oryginalną ramkę do detekcji
        with last_frame_lock:
            last_frame = frame.copy()
        
        # Zmień rozmiar do 224x224 (standardowy input dla ViViT)
        frame_resized = cv2.resize(frame, (224, 224))
        
        with buffer_lock:
            frame_buffer.append(frame_resized)
        
        return True
    except Exception as e:
        print(f"[!] Blad przetwarzania ramki: {e}")
        return False


def generate_video_description(frame):
    """Generuje opis klatki wideo za pomocą InternVL"""
    try:
        if frame is None or internvl_model is None:
            return "Model niedostępny"
        
        # Konwertuj numpy array do PIL Image
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(frame)
        else:
            frame_pil = frame
        
        # Przetwórz obraz dla InternVL
        pixel_values = preprocess_image_for_internvl(frame_pil, input_size=448, max_num=4)
        
        # Przenieś na odpowiednie urządzenie
        if device == "cuda":
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
        else:
            pixel_values = pixel_values.to(torch.float32)
        
        # Przygotuj prompt z tagiem <image>
        question = "<image>\nDescribe briefly."
        
        # Konfiguracja generowania - szybka
        generation_config = dict(
            max_new_tokens=100,
            do_sample=False,
        )
        
        # Generuj odpowiedź
        with torch.no_grad():
            response = internvl_model.chat(
                internvl_tokenizer,
                pixel_values,
                question,
                generation_config
            )
        
        return response if response else "Brak opisu"
    except Exception as e:
        print(f"[!] Blad generowania opisu InternVL: {e}")
        return f"Błąd"


def classify_image_vit(frame):
    """Klasyfikuje pojedynczą klatkę za pomocą modelu ViT"""
    try:
        if frame is None or vit_model is None:
            return []
        
        # Przygotuj input
        inputs = vit_processor(images=frame, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = vit_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            top_probs, top_indices = torch.topk(probs, 3)
        
        vit_classes = []
        for i in range(min(3, len(top_probs))):
            idx = int(top_indices[i].cpu().item())
            prob = float(top_probs[i].cpu().item())
            class_name = vit_model.config.id2label.get(idx, f"Class {idx}")
            
            vit_classes.append({
                "class": class_name,
                "confidence": prob
            })
        
        return vit_classes
    except Exception as e:
        print(f"[!] Blad klasyfikacji ViT: {e}")
        return []


def detect_objects(frame):
    """Wykrywa obiekty w ramce za pomoca modelu RT-DETR"""
    try:
        if frame is None or detection_model is None:
            return []
        
        # Przygotuj input
        inputs = detection_processor(images=frame, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = detection_model(**inputs)
        
        # Przetwórz wyniki (RT-DETR format)
        # frame.shape[:2] to (height, width), ale potrzebujemy (width, height) czyli [::-1]
        target_sizes = torch.tensor([frame.shape[:2][::-1]])
        results = detection_processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=0.3
        )
        
        objects = []
        if len(results) > 0:
            result = results[0]
            scores = result.get("scores", torch.tensor([]))
            labels = result.get("labels", torch.tensor([]))
            
            for score, label_id in zip(scores, labels):
                score_val = score.item() if hasattr(score, 'item') else float(score)
                label_val = label_id.item() if hasattr(label_id, 'item') else int(label_id)
                
                # Pobierz nazwę klasy z konfiguracji modelu
                try:
                    label_name = detection_model.config.id2label.get(label_val, f"Object {label_val}")
                except:
                    label_name = f"Object {label_val}"
                
                objects.append({
                    "name": label_name,
                    "confidence": float(score_val)
                })
        
        return objects[:5]  # Zwróć maksymalnie 5 top obiektów
    except Exception as e:
        print(f"[!] Blad detekcji obiektow: {e}")
        return []


def run_inference():
    """Uruchamia inference gdy bufor jest pełny"""
    global last_prediction
    
    while True:
        try:
            with buffer_lock:
                if len(frame_buffer) == FRAME_BUFFER_SIZE:
                    # Skopiuj ramki z bufora
                    frames = list(frame_buffer)
                else:
                    time.sleep(0.1)
                    continue
            
            # Przygotuj dane dla modelu
            frames_np = np.array(frames)
            
            # Przetwórz przez processor
            inputs = processor(list(frames_np), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Pobierz probabilności wszystkich klas
                probs = torch.softmax(logits, dim=-1)[0]
                top_probs, top_indices = torch.topk(probs, 5)
                
                # Dynamicznie określ ile klas pokazać na podstawie pewności
                best_confidence = top_probs[0].item()
                
                # Jeśli najlepsza klasa ma >60% pewności, pokazuj tylko 1
                # Jeśli ma <60%, pokazuj 3 żeby pokazać alternatywy
                if best_confidence > 0.6:
                    num_to_show = 1
                elif best_confidence > 0.4:
                    num_to_show = 2
                else:
                    num_to_show = 3
                
                # Przygotuj listę top klas
                top_classes = []
                for i in range(num_to_show):
                    idx = int(top_indices[i].cpu().item())
                    prob = float(top_probs[i].cpu().item())
                    class_name = model.config.id2label.get(idx, f"Unknown class {idx}")
                    
                    # Jeśli etykieta to LABEL_XXX, spróbuj użyć indeksu jako opisu
                    if class_name.startswith("LABEL_"):
                        class_name = f"Activity {idx} (label not loaded)"
                    
                    top_classes.append({
                        "class": class_name,
                        "confidence": prob
                    })
                
                # Wykryj obiekty i sklasyfikuj pojedynczą klatkę
                objects = []
                vit_classes = []
                with last_frame_lock:
                    if last_frame is not None:
                        objects = detect_objects(last_frame)
                        vit_classes = classify_image_vit(last_frame)
                
                # Opis sceny jest generowany tylko na żądanie (klik w UI)
                video_description = last_prediction.get("video_description", "Oczekiwanie...")
                
                with prediction_lock:
                    last_prediction = {
                        "top_classes": top_classes,
                        "objects": objects,
                        "vit_classes": vit_classes,
                        "video_description": video_description
                    }
                
                top_classes_str = ", ".join([f"{c['class']} ({c['confidence']:.0%})" for c in top_classes[:3]])
                objects_str = ", ".join([f"{o['name']} ({o['confidence']:.0%})" for o in objects]) if objects else "brak"
                vit_str = ", ".join([f"{c['class']} ({c['confidence']:.0%})" for c in vit_classes[:2]]) if vit_classes else "brak"
                desc_str = video_description[:80] + "..." if len(video_description) > 80 else video_description
                print(f"[PRED] Top akcje: {top_classes_str} | Obiekty: {objects_str} | ViT: {vit_str}")
                print(f"[INTERNVL] {desc_str}")
            
            time.sleep(1.2)  # Zwolniona pętla inference dla mniejszego obciążenia
            
        except Exception as e:
            print(f"[!] Blad podczas inference: {e}")
            time.sleep(1)


# Uruchom wątek inference w tle
inference_thread = threading.Thread(target=run_inference, daemon=True)
inference_thread.start()


@app.route('/')
def index():
    """Strona główna z interfejsem"""
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    """Obsługiwanie połączenia WebSocket"""
    print(f"[WS] Nowy klient połączony: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    """Obsługiwanie rozłączenia WebSocket"""
    print(f"[WS] Klient rozłączony: {request.sid}")


@socketio.on('send_frame')
def handle_frame(data):
    """Odbieranie ramki przez WebSocket"""
    try:
        frame_data = data.get('frame')
        
        if not frame_data:
            emit('error', {'message': 'Brak danych ramki'})
            return
        
        # Przetwórz ramkę
        success = process_frame(frame_data)
        
        if not success:
            emit('error', {'message': 'Błąd przetwarzania ramki'})
            return
        
        # Wyślij aktualną predykcję
        with prediction_lock:
            prediction = last_prediction.copy()
        
        emit('prediction_update', prediction)
    
    except Exception as e:
        emit('error', {'message': str(e)})


@socketio.on('request_description')
def handle_request_description():
    """Generuje opis sceny na żądanie (klik w UI)"""
    try:
        with last_frame_lock:
            frame = None if last_frame is None else last_frame.copy()
        if frame is None:
            emit('description_response', {'video_description': 'Brak ramki do opisu'})
            return
        if internvl_model is None:
            emit('description_response', {'video_description': 'Model niedostępny'})
            return

        desc = generate_video_description(frame)
        with prediction_lock:
            last_prediction['video_description'] = desc
        emit('description_response', {'video_description': desc})
    except Exception as e:
        emit('description_response', {'video_description': f'Błąd: {e}'})


@app.route('/api/prediction', methods=['GET'])
def get_prediction():
    """Endpoint do pobierania ostatniej predykcji"""
    with prediction_lock:
        prediction = last_prediction.copy()
    return jsonify(prediction)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "device": device,
        "model": MODEL_NAME,
        "buffer_size": len(frame_buffer)
    })


if __name__ == '__main__':
    enable_https = os.getenv('ENABLE_HTTPS', 'false').lower() in ['1', 'true', 'yes']
    cert_path = os.getenv('SSL_CERT_PATH')
    key_path = os.getenv('SSL_KEY_PATH')

    print("\n" + "="*50)
    print("Serwer ViViT uruchomiony!")
    print(f"Urządzenie: {device}")

    ssl_context = None
    if enable_https:
        if cert_path and key_path and os.path.exists(cert_path) and os.path.exists(key_path):
            ssl_context = (cert_path, key_path)
            print("Tryb: HTTPS (lokalny certyfikat)")
            print(f"Cert: {cert_path}")
            print(f"Klucz: {key_path}")
            print("Otwórz https://localhost:5000 w przeglądarce")
        else:
            print("UWAGA: ENABLE_HTTPS=true, ale brak poprawnych ścieżek do certyfikatu/klucza.")
            print("Używam HTTP.")
            print("Otwórz http://localhost:5000 w przeglądarce")
    else:
        print("Tryb: HTTP")
        print("Otwórz http://localhost:5000 w przeglądarce")

    print("="*50 + "\n")

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, ssl_context=ssl_context)
