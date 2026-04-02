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

# Constants for InternVL
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """Builds image transform for InternVL"""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Finds the closest aspect ratio for InternVL"""
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
    """Dynamic image preprocessing for InternVL"""
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
    """Preprocesses a PIL image for InternVL"""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ViViT model configuration
MODEL_NAME = "google/vivit-b-16x2-kinetics400"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading ViViT model on device: {device}")
processor = VivitImageProcessor.from_pretrained(MODEL_NAME)
model = VivitForVideoClassification.from_pretrained(
    MODEL_NAME,
    ignore_mismatched_sizes=False,
)

# Load Kinetics-400 labels
print("[*] Loading Kinetics-400 labels from local file...")
try:
    with open("kinetics400_labels.json", "r") as f:
        labels_data = json.load(f)
        id2label = {int(k): v for k, v in labels_data.items()}
        model.config.id2label = id2label
        model.config.label2id = {v: k for k, v in id2label.items()}
        print(f"[OK] Loaded {len(id2label)} class labels")
except Exception as e:
    print(f"[!] Error loading labels: {e}")
    model.config.id2label = {i: f"Activity {i}" for i in range(400)}

model = model.to(device)
model.eval()
print("ViViT model loaded!")
sample_label = model.config.id2label.get(0, "Unknown")
print(f"Sample class: {sample_label}")

# RT-DETR model configuration (object detection)
print(f"\n[*] Loading RT-DETR model on device: {device}")
try:
    DETECTION_MODEL_NAME = "PekingU/rtdetr_r50vd"
    detection_processor = RTDetrImageProcessor.from_pretrained(DETECTION_MODEL_NAME)
    detection_model = RTDetrForObjectDetection.from_pretrained(DETECTION_MODEL_NAME)
    detection_model = detection_model.to(device)
    detection_model.eval()
    print("[OK] RT-DETR model loaded!")
except Exception as e:
    print(f"[!] Error loading detection model: {e}")
    print("[*] Object detection disabled")
    detection_model = None
    detection_processor = None

# ViT model configuration (single image classification)
print(f"\n[*] Loading ViT model on device: {device}")
try:
    VIT_MODEL_NAME = "google/vit-base-patch16-224"
    vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
    vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME)
    vit_model = vit_model.to(device)
    vit_model.eval()
    print(f"[OK] ViT model loaded! Classes: {len(vit_model.config.id2label)}")
except Exception as e:
    print(f"[!] Error loading ViT model: {e}")
    print("[*] Image classification disabled")
    vit_model = None
    vit_processor = None

# InternVL3_5-4B model configuration (Vision-Language Model)
print(f"\n[*] Loading InternVL3_5-4B model on device: {device}")
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
    print(f"[OK] InternVL3_5-4B-Flash model loaded!")
except Exception as e:
    print(f"[!] Error loading InternVL model: {e}")
    print("[*] Video description generation disabled")
    internvl_model = None
    internvl_tokenizer = None

# Frame buffer for video (ViViT requires a sequence of frames)
FRAME_BUFFER_SIZE = 32  # ViViT uses 32 frames by default
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
buffer_lock = threading.Lock()

# Cache for the last prediction
last_prediction = {
    "top_classes": [
        {"class": "Waiting...", "confidence": 0.0}
    ],
    "objects": [],
    "vit_classes": [],
    "video_description": "Waiting..."
}
prediction_lock = threading.Lock()

# Cache for the last frame for detection
last_frame = None
last_frame_lock = threading.Lock()


def process_frame(frame_base64):
    """Decodes a frame from base64 and adds it to the buffer"""
    global last_frame
    try:
        img_data = base64.b64decode(frame_base64.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save the original frame for detection
        with last_frame_lock:
            last_frame = frame.copy()
        
        # Resize to 224x224 (standard ViViT input size)
        frame_resized = cv2.resize(frame, (224, 224))
        
        with buffer_lock:
            frame_buffer.append(frame_resized)
        
        return True
    except Exception as e:
        print(f"[!] Error processing frame: {e}")
        return False


def generate_video_description(frame):
    """Generates a description of the video frame using InternVL"""
    try:
        if frame is None or internvl_model is None:
            return "Model unavailable"
        
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(frame)
        else:
            frame_pil = frame
        
        pixel_values = preprocess_image_for_internvl(frame_pil, input_size=448, max_num=4)
        
        # Move to the appropriate device
        if device == "cuda":
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
        else:
            pixel_values = pixel_values.to(torch.float32)
        
        question = "<image>\nDescribe briefly."
        
        # Fast generation config
        generation_config = dict(
            max_new_tokens=100,
            do_sample=False,
        )
        
        with torch.no_grad():
            response = internvl_model.chat(
                internvl_tokenizer,
                pixel_values,
                question,
                generation_config
            )
        
        return response if response else "No description"
    except Exception as e:
        print(f"[!] Error generating InternVL description: {e}")
        return "Error"


def classify_image_vit(frame):
    """Classifies a single frame using the ViT model"""
    try:
        if frame is None or vit_model is None:
            return []
        
        inputs = vit_processor(images=frame, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
        print(f"[!] Error during ViT classification: {e}")
        return []


def detect_objects(frame):
    """Detects objects in a frame using the RT-DETR model"""
    try:
        if frame is None or detection_model is None:
            return []
        
        inputs = detection_processor(images=frame, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = detection_model(**inputs)
        
        # Post-process RT-DETR results; frame.shape[:2] is (height, width), reversed to (width, height)
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
                
                try:
                    label_name = detection_model.config.id2label.get(label_val, f"Object {label_val}")
                except:
                    label_name = f"Object {label_val}"
                
                objects.append({
                    "name": label_name,
                    "confidence": float(score_val)
                })
        
        return objects[:5]  # Return at most 5 top objects
    except Exception as e:
        print(f"[!] Error during object detection: {e}")
        return []


def run_inference():
    """Runs inference when the buffer is full"""
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
            
            inputs = processor(list(frames_np), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                probs = torch.softmax(logits, dim=-1)[0]
                top_probs, top_indices = torch.topk(probs, 5)
                
                # Dynamically determine how many classes to show based on confidence:
                # >60% → show 1, >40% → show 2, otherwise show 3
                best_confidence = top_probs[0].item()
                
                if best_confidence > 0.6:
                    num_to_show = 1
                elif best_confidence > 0.4:
                    num_to_show = 2
                else:
                    num_to_show = 3
                
                top_classes = []
                for i in range(num_to_show):
                    idx = int(top_indices[i].cpu().item())
                    prob = float(top_probs[i].cpu().item())
                    class_name = model.config.id2label.get(idx, f"Unknown class {idx}")
                    
                    # If the label is LABEL_XXX, fall back to index-based description
                    if class_name.startswith("LABEL_"):
                        class_name = f"Activity {idx} (label not loaded)"
                    
                    top_classes.append({
                        "class": class_name,
                        "confidence": prob
                    })
                
                # Detect objects and classify the current frame
                objects = []
                vit_classes = []
                with last_frame_lock:
                    if last_frame is not None:
                        objects = detect_objects(last_frame)
                        vit_classes = classify_image_vit(last_frame)
                
                # Scene description is generated on demand only (UI click)
                video_description = last_prediction.get("video_description", "Waiting...")
                
                with prediction_lock:
                    last_prediction = {
                        "top_classes": top_classes,
                        "objects": objects,
                        "vit_classes": vit_classes,
                        "video_description": video_description
                    }
                
                top_classes_str = ", ".join([f"{c['class']} ({c['confidence']:.0%})" for c in top_classes[:3]])
                objects_str = ", ".join([f"{o['name']} ({o['confidence']:.0%})" for o in objects]) if objects else "none"
                vit_str = ", ".join([f"{c['class']} ({c['confidence']:.0%})" for c in vit_classes[:2]]) if vit_classes else "none"
                desc_str = video_description[:80] + "..." if len(video_description) > 80 else video_description
                print(f"[PRED] Top actions: {top_classes_str} | Objects: {objects_str} | ViT: {vit_str}")
                print(f"[INTERNVL] {desc_str}")
            
            time.sleep(1.2)  # Slowed inference loop to reduce load
            
        except Exception as e:
            print(f"[!] Error during inference: {e}")
            time.sleep(1)


# Start inference thread in the background
inference_thread = threading.Thread(target=run_inference, daemon=True)
inference_thread.start()


@app.route('/')
def index():
    """Main page with the UI"""
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    """Handles WebSocket connection"""
    print(f"[WS] New client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    """Handles WebSocket disconnection"""
    print(f"[WS] Client disconnected: {request.sid}")


@socketio.on('send_frame')
def handle_frame(data):
    """Receives a frame via WebSocket"""
    try:
        frame_data = data.get('frame')
        
        if not frame_data:
            emit('error', {'message': 'No frame data'})
            return
        
        success = process_frame(frame_data)
        
        if not success:
            emit('error', {'message': 'Frame processing error'})
            return
        
        with prediction_lock:
            prediction = last_prediction.copy()
        
        emit('prediction_update', prediction)
    
    except Exception as e:
        emit('error', {'message': str(e)})


@socketio.on('request_description')
def handle_request_description():
    """Generates a scene description on demand (UI click)"""
    try:
        with last_frame_lock:
            frame = None if last_frame is None else last_frame.copy()
        if frame is None:
            emit('description_response', {'video_description': 'No frame available'})
            return
        if internvl_model is None:
            emit('description_response', {'video_description': 'Model unavailable'})
            return

        desc = generate_video_description(frame)
        with prediction_lock:
            last_prediction['video_description'] = desc
        emit('description_response', {'video_description': desc})
    except Exception as e:
        emit('description_response', {'video_description': f'Error: {e}'})


@app.route('/api/prediction', methods=['GET'])
def get_prediction():
    """Endpoint for retrieving the latest prediction"""
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
    print("ViViT server started!")
    print(f"Device: {device}")

    ssl_context = None
    if enable_https:
        if cert_path and key_path and os.path.exists(cert_path) and os.path.exists(key_path):
            ssl_context = (cert_path, key_path)
            print("Mode: HTTPS (local certificate)")
            print(f"Cert: {cert_path}")
            print(f"Key: {key_path}")
            print("Open https://localhost:5000 in your browser")
        else:
            print("WARNING: ENABLE_HTTPS=true, but no valid certificate/key paths provided.")
            print("Using HTTP.")
            print("Open http://localhost:5000 in your browser")
    else:
        print("Mode: HTTP")
        print("Open http://localhost:5000 in your browser")

    print("="*50 + "\n")

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, ssl_context=ssl_context)
