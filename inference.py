import os
import shutil
import torch
import onnxruntime
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric

# Function to calculate IoU
def calculate_iou(pred, target, num_classes):
    iou_scores = []
    pred = pred.argmax(dim=0).flatten()
    target = target.flatten()
    
    for cls in range(num_classes):
        intersection = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)
    
    return iou_scores

# Convert torch to onnx model
dummy_input = torch.randn(1, 4, 240, 240, 160).to(device)
onnx_path = os.path.join(root_dir, "best_metric_model.onnx")
torch.onnx.export(model, dummy_input, onnx_path, verbose=False)

# Inference function
def onnx_infer(inputs):
    ort_inputs = {ort_session.get_inputs()[0].name: inputs.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    return torch.Tensor(ort_outs[0]).to(inputs.device)

def predict(input):
    return sliding_window_inference(
        inputs=input,
        roi_size=(240, 240, 160),
        sw_batch_size=1,
        predictor=onnx_infer,
        overlap=0.5,
    )

# Load the ONNX model
onnx_model_path = os.path.join(root_dir, "best_metric_model.onnx")
ort_session = onnxruntime.InferenceSession(onnx_model_path)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

# Evaluate the model on the validation dataset
for val_data in tqdm(val_loader, desc="Onnxruntime Inference Progress"):
    val_inputs, val_labels = (
        val_data["image"].to(device),
        val_data["label"].to(device),
    )

    ort_outs = predict(val_inputs)
    val_outputs = post_trans(torch.Tensor(ort_outs[0]).to(device)).unsqueeze(0)

    # Calculate IoU for each class
    iou_scores = calculate_iou(val_outputs, val_labels, num_classes=3)  # Adjust num_classes as needed
    print(f"IoU scores for current batch: {iou_scores}")

    # Update metrics
    dice_metric(y_pred=val_outputs, y=val_labels)
    dice_metric_batch(y_pred=val_outputs, y=val_labels)

# Calculate and print overall metrics
onnx_metric = dice_metric.aggregate().item()
onnx_metric_batch = dice_metric_batch.aggregate()
onnx_metric_tc = onnx_metric_batch[0].item()
onnx_metric_wt = onnx_metric_batch[1].item()
onnx_metric_et = onnx_metric_batch[2].item()

print(f"onnx metric: {onnx_metric}")
print(f"onnx_metric_tc: {onnx_metric_tc:.4f}")
print(f"onnx_metric_wt: {onnx_metric_wt:.4f}")
print(f"onnx_metric_et: {onnx_metric_et:.4f}")

# Visualize a sample image, its label, and the model outputs
with torch.no_grad():
    val_input = val_ds[6]["image"].unsqueeze(0).to(device)
    val_output = predict(val_input)
    ort_output = predict(val_input)
    
    val_output = post_trans(val_output[0])
    ort_output = post_trans(torch.Tensor(ort_output[0]).to(device)).unsqueeze(0)
    
    # Visualize images and outputs
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(val_ds[6]["image"][i, :, :, 70].detach().cpu(), cmap="gray")
    plt.show()
    
    plt.figure("label", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(val_ds[6]["label"][i, :, :, 70].detach().cpu())
    plt.show()
    
    plt.figure("output", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"pth output channel {i}")
        plt.imshow(val_output[i, :, :, 70].detach().cpu())
    plt.show()
    
    plt.figure("onnx output", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"onnx output channel {i}")
        plt.imshow(ort_output[0, i, :, :, 70].detach().cpu())
    plt.show()

# Cleanup data directory
if directory is None:
    shutil.rmtree(root_dir)
