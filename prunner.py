import torch

from yolov6.utils.config import Config
from yolov6.models.yolo import build_model
from yolov6.utils.checkpoint import load_state_dict
from yolov6.utils.envs import select_device

gates = torch.load("runs/inference/exp1/gates.pt")

print(gates[0])

def get_model(nc):
    device = select_device("cpu")
    cfg = Config.fromfile("configs/simplified/G_yolov6m_SIM_optimized.py")
    model = build_model(cfg, nc, device)
    weights = "runs/train/Medium-VOC[Gated]9/weights/best_ckpt.pt"
    if weights:
        model = load_state_dict(weights, model, map_location=device)

    return model

model = get_model(20)

print(model.weights)
