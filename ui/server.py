"""VocAssist inference server.

Usage:
    # from project root:
    python ui/server.py

Requires:
    pip install flask

Endpoints:
    GET  /              → serves index.html
    POST /api/inference → multipart image → JSON with bbox/gradcam images + metrics
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.voc import VOC_CLASSES, NUM_CLASSES, NUM_CLASSES_PERSON
from models.YOLO import YOLOv8
from utils.detection import decode_predictions
from utils.gradcam import GradCAM, get_target_layer

# ── Flask app — serve static files from ui/ directory ────────────────────────
UI_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=UI_DIR, static_url_path='')

# ── Globals set at startup ────────────────────────────────────────────────────
DEVICE        = torch.device('cpu')
YOLO_MODEL    = None
RESNET_MODEL  = None
RESNET_CLASSES: list[str] = []
PERSON_ONLY   = True          # set by --person_only flag at startup
YOLO_CLASS_NAMES: list[str] = ['person']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

YOLO_TRANSFORM = T.Compose([
    T.Resize((320, 320)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

RESNET_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(UI_DIR, 'index.html')


@app.route('/api/inference', methods=['POST'])
def inference():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    pil_orig = Image.open(request.files['image'].stream).convert('RGB')

    result = {
        'success':            True,
        'boxes':              [],
        'bbox_image_b64':     None,
        'gradcam_b64':        None,
        'top_class':          '—',
        'top_confidence':     0.0,
        'yolo_confidence':    0.0,
        'resnet_confidence':  None,
        'ensemble_confidence':0.0,
    }

    # ── YOLO ─────────────────────────────────────────────────────────────────
    if YOLO_MODEL is not None:
        img_t = YOLO_TRANSFORM(pil_orig).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = YOLO_MODEL(img_t)
            raw     = YOLO_MODEL.decode(outputs, img_size=320, conf_thresh=0.25)
        preds = decode_predictions(raw.cpu(), conf_thresh=0.25, iou_thresh=0.45)
        det   = preds[0]

        boxes_out = []
        for i in range(len(det.get('scores', []))):
            score = float(det['scores'][i])
            label = int(det['labels'][i])
            x1, y1, x2, y2 = det['boxes'][i].tolist()
            boxes_out.append({
                'x1': round(x1 / 320, 4), 'y1': round(y1 / 320, 4),
                'x2': round(x2 / 320, 4), 'y2': round(y2 / 320, 4),
                'class_name': YOLO_CLASS_NAMES[label] if label < len(YOLO_CLASS_NAMES) else str(label),
                'confidence': round(score, 3),
            })

        result['boxes']         = boxes_out
        result['bbox_image_b64'] = _draw_bbox_image(pil_orig, boxes_out)

        if boxes_out:
            best = max(boxes_out, key=lambda b: b['confidence'])
            result['top_class']     = best['class_name']
            result['yolo_confidence'] = best['confidence']

    # ── ResNet + GradCAM ──────────────────────────────────────────────────────
    if RESNET_MODEL is not None:
        img_t = RESNET_TRANSFORM(pil_orig).unsqueeze(0).to(DEVICE)

        cam = GradCAM(RESNET_MODEL, get_target_layer(RESNET_MODEL))
        heatmap, pred_cls = cam.generate(img_t)
        cam.remove_hooks()

        with torch.no_grad():
            logits = RESNET_MODEL(img_t)
        probs = torch.softmax(logits, dim=1)[0]
        # For binary classifier always report person-class probability (class 1)
        person_idx = 1 if len(RESNET_CLASSES) == 2 else pred_cls
        conf = float(probs[person_idx])

        result['resnet_confidence'] = round(conf, 3)
        if result['top_class'] == '—' and conf >= 0.5:
            result['top_class'] = 'person'

        result['gradcam_b64'] = _overlay_gradcam(pil_orig, heatmap)

    # ── Ensemble score (mean of available confidences) ────────────────────────
    confs = [v for k, v in result.items() if k.endswith('_confidence') and v > 0]
    result['ensemble_confidence'] = round(sum(confs) / len(confs), 3) if confs else 0.0

    return jsonify(result)


# ── Image helpers ─────────────────────────────────────────────────────────────

def _denorm_tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """(3, H, W) normalised tensor → PIL RGB image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (img_tensor * std + mean).clamp(0, 1)
    return Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def _draw_bbox_image(pil_orig: Image.Image, boxes: list[dict]) -> str:
    """Draw teal bounding boxes on the original image using normalised [0,1] coords."""
    pil = pil_orig.copy()
    w, h = pil.size

    if boxes:
        draw = ImageDraw.Draw(pil)
        teal = (29, 233, 182)
        dark = (13, 27, 42)

        for box in boxes:
            x1 = box['x1'] * w
            y1 = box['y1'] * h
            x2 = box['x2'] * w
            y2 = box['y2'] * h

            draw.rectangle([x1, y1, x2, y2], outline=teal, width=3)

            label = f"{box['class_name']}  {box['confidence']:.2f}"
            lw    = len(label) * 7 + 6
            lh    = 16
            ly    = max(y1 - lh, 0)
            draw.rectangle([x1, ly, x1 + lw, ly + lh], fill=teal)
            draw.text((x1 + 3, ly + 2), label, fill=dark)

    return _pil_to_b64(pil)


def _overlay_gradcam(pil_orig: Image.Image, heatmap_np: np.ndarray) -> str:
    """Blend a Grad-CAM heatmap (0-1) onto the original image."""
    import matplotlib.cm as cm

    img_np = np.array(pil_orig.convert('RGB'), dtype=np.float32)
    h, w   = img_np.shape[:2]

    hm_pil = Image.fromarray((heatmap_np * 255).astype(np.uint8))
    hm_pil = hm_pil.resize((w, h), Image.BILINEAR)
    hm_np  = np.array(hm_pil).astype(np.float32) / 255.0

    hm_colored = (cm.jet(hm_np)[:, :, :3] * 255).astype(np.float32)
    overlay    = (0.55 * img_np + 0.45 * hm_colored).clip(0, 255).astype(np.uint8)

    return _pil_to_b64(Image.fromarray(overlay))


def _pil_to_b64(pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(args) -> None:
    global YOLO_MODEL, RESNET_MODEL, RESNET_CLASSES, DEVICE, PERSON_ONLY, YOLO_CLASS_NAMES

    if args.device == 'auto':
        DEVICE = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps'  if torch.backends.mps.is_available() else 'cpu'
        )
    else:
        DEVICE = torch.device(args.device)
    print(f'Device: {DEVICE}')

    PERSON_ONLY = args.person_only

    # YOLO
    yolo_path = os.path.join(os.path.dirname(UI_DIR), args.yolo_path) \
                if not os.path.isabs(args.yolo_path) else args.yolo_path
    if args.yolo_path and os.path.exists(yolo_path):
        n_cls = NUM_CLASSES_PERSON if args.person_only else NUM_CLASSES
        YOLO_CLASS_NAMES = ['person'] if args.person_only else list(VOC_CLASSES)
        m = YOLOv8(num_classes=n_cls).to(DEVICE)
        m.load_state_dict(torch.load(yolo_path, map_location=DEVICE))
        m.eval()
        YOLO_MODEL = m
        print(f'YOLO loaded:   {yolo_path}  ({"person-only" if args.person_only else "20 classes"})')
    else:
        print(f'YOLO not found ({args.yolo_path}) — detection disabled')

    # ResNet
    resnet_path = os.path.join(os.path.dirname(UI_DIR), args.resnet_path) \
                  if args.resnet_path and not os.path.isabs(args.resnet_path) else args.resnet_path
    if args.resnet_path and os.path.exists(resnet_path):
        n_cls = args.resnet_classes
        if args.resnet_arch == 'pretrained':
            import torchvision.models as tvm
            m = tvm.resnet18(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, n_cls)
        else:
            from models.ResNet import ResNet, BasicBlock
            m = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=n_cls)
        m.load_state_dict(torch.load(resnet_path, map_location=DEVICE))
        m.to(DEVICE).eval()
        RESNET_MODEL = m
        if n_cls == 2:
            RESNET_CLASSES = ['no_person', 'person']
        elif n_cls == 10:
            RESNET_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                              'dog', 'frog', 'horse', 'ship', 'truck']
        elif n_cls <= 20:
            RESNET_CLASSES = list(VOC_CLASSES[:n_cls])
        else:
            RESNET_CLASSES = [str(i) for i in range(n_cls)]
        print(f'ResNet loaded: {resnet_path}  ({n_cls} classes, arch={args.resnet_arch})')
    else:
        print(f'ResNet not found ({args.resnet_path}) — GradCAM disabled')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='VocAssist inference server')
    p.add_argument('--yolo_path',      type=str, default='weights/yolo_best.pth')
    p.add_argument('--resnet_path',    type=str, default='weights/best_model_person.pth')
    p.add_argument('--resnet_arch',    type=str, default='pretrained',
                   choices=['custom', 'pretrained'])
    p.add_argument('--resnet_classes', type=int, default=2)
    p.add_argument('--person_only',    action=argparse.BooleanOptionalAction, default=True,
                   help='Load YOLO as person-only (1 class) checkpoint (default: True)')
    p.add_argument('--device',         type=str, default='auto')
    p.add_argument('--port',           type=int, default=5000)
    p.add_argument('--debug',          action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    load_models(args)
    print(f'\nVocAssist →  http://localhost:{args.port}\n')
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)