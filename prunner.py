import torch
import os

from yolov6.utils.config import Config
from yolov6.models.yolo import build_model
from yolov6.utils.checkpoint import load_state_dict
from yolov6.utils.envs import select_device
from yolov6.utils.ema import ModelEMA, de_parallel

from copy import deepcopy


# Load gating decisions
gates = torch.load("runs/inference/exp30/gates.pt")

def apply_gating_decisions(model, gates, device):
    """
    Apply gating decisions to the model. Assumes 'gates' is a list with each element
    corresponding to a gating decision for each applicable layer.
    """
    i = -1

    for _, (name, param) in enumerate(model.named_parameters()):
        # Ensure we do not exceed the gates list.
        if i >= len(gates):
            continue

        if not (("conv.weight" in name and "rbr_dense" not in name and "rbr_1x1.conv.weight" not in name) or ("rbr_1x1.bn.weight" in name) or ("upsample_transpose.weight" in name) or ("reg_preds" in name and "weight" in name) or ("cls_preds" in name and "weight" in name)):
            continue
        else:
            i += 1

        gating_decision = gates[i][0]

        print("##### Processing ###### ", i)

        # Handle the None case directly
        if gating_decision is None:
            param.data.zero_()
            continue

        gating_decision = gating_decision.to(device)

        try:
            # For lists or individual numerics, ensure conversion to tensor is valid
            if isinstance(gating_decision, (list, int, float)):
                gating_decision = [gating_decision] if isinstance(gating_decision, (int, float)) else gating_decision
                gating_tensor = torch.tensor(gating_decision, device=param.device, dtype=param.dtype)
            elif isinstance(gating_decision, torch.Tensor):
                gating_tensor = gating_decision
            else:
                raise TypeError(f"Unsupported gating decision type for layer {name}: {type(gating_decision)}")

            if(len(param.data.shape) > 1):
                gating_tensor = gating_tensor.view(-1, 1, 1, 1)
            else:
                gating_tensor = gating_tensor.view(-1)

            if "detect." in name:
                continue

            # Check for dimension compatibility
            if gating_tensor.size(0) == param.data.size(0):
                param.data.mul_(gating_tensor)
            else:
                print(f"Dimension mismatch for layer {name}. Param size: {param.data.size(0)}, gating size: {gating_tensor.size(0)}")

        except TypeError as e:
            print(f"Failed to apply gating to layer {name}: {e}")


def get_model_and_save_pruned(nc, weights_path):
    device = select_device("cpu")
    cfg = Config.fromfile("configs/simplified/G_yolov6m_SIM_optimized.py")
    model = build_model(cfg, nc, device)
    if weights_path:
        ckpt = torch.load(weights, map_location=device)
        model = load_state_dict(weights_path, model, map_location=device)
        for name, param in model.named_parameters():
            print(name, param.size(), param.numel())
    
    # Zero out channels based on gates
    apply_gating_decisions(model, gates, device)
    for name, param in model.named_parameters():
        print(name, param.size(), param.numel())
    
    # Saving the modified model with "_pruned" suffix
    pruned_weights_path = os.path.splitext(weights_path)[0] + "_pruned.pt"

    ckpt = {
            'model': deepcopy(de_parallel(model)),
            'ema': ckpt['ema'],
            'updates': ckpt['updates'],
            'optimizer': ckpt['optimizer'],
            'scheduler': ckpt['scheduler'],
            'epoch': ckpt['epoch'],
            'results': ckpt['results'],
            }

    torch.save(ckpt, pruned_weights_path)
    print(f"Pruned model saved to: {pruned_weights_path}")

    return model

# Assuming 'weights' contains the path to the original model weights
weights = "weights/G-Medium.pt"
model = get_model_and_save_pruned(20, weights)
