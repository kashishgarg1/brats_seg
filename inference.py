"""## Convert torch to onnx model"""

dummy_input = torch.randn(1, 4, 240, 240, 160).to(device)
onnx_path = os.path.join(root_dir, "best_metric_model.onnx")
torch.onnx.export(model, dummy_input, onnx_path, verbose=False)

"""## Inference onnx model
Here we change the model used by predictor to onnx_infer, both of which are used to obtain a tensor after the input has been reasoned by the neural network.

Note: If the warning `pthread_setaffinity_np failed` appears when executing this cell, this is a known problem with the onnxruntime and does not affect the execution result. If you want to disable the warning, you can cancel the following comment to solve the problem.
"""

# Using the following program snippet will not affect the execution time.
# options = ort.SessionOptions()
# options.intra_op_num_threads = 1
# options.inter_op_num_threads = 1

def onnx_infer(inputs):
    ort_inputs = {ort_session.get_inputs()[0].name: inputs.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    return torch.Tensor(ort_outs[0]).to(inputs.device)


def predict(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=onnx_infer,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

onnx_model_path = os.path.join(root_dir, "best_metric_model.onnx")
ort_session = onnxruntime.InferenceSession(onnx_model_path)

for val_data in tqdm(val_loader, desc="Onnxruntime Inference Progress"):
    val_inputs, val_labels = (
        val_data["image"].to(device),
        val_data["label"].to(device),
    )

    ort_outs = predict(val_inputs)
    val_outputs = post_trans(torch.Tensor(ort_outs[0]).to(device)).unsqueeze(0)

    dice_metric(y_pred=val_outputs, y=val_labels)
    dice_metric_batch(y_pred=val_outputs, y=val_labels)
onnx_metric = dice_metric.aggregate().item()
onnx_metric_batch = dice_metric_batch.aggregate()
onnx_metric_tc = onnx_metric_batch[0].item()
onnx_metric_wt = onnx_metric_batch[1].item()
onnx_metric_et = onnx_metric_batch[2].item()

print(f"onnx metric: {onnx_metric}")
print(f"onnx_metric_tc: {onnx_metric_tc:.4f}")
print(f"onnx_metric_wt: {onnx_metric_wt:.4f}")
print(f"onnx_metric_et: {onnx_metric_et:.4f}")

"""## Check best onnx model output with the input image and label"""

onnx_model_path = os.path.join(root_dir, "best_metric_model.onnx")
ort_session = onnxruntime.InferenceSession(onnx_model_path)
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()

with torch.no_grad():
    # select one image to evaluate and visualize the model output
    val_input = val_ds[6]["image"].unsqueeze(0).to(device)
    val_output = inference(val_input)
    val_output = post_trans(val_output[0])
    ort_output = predict(val_input)
    ort_output = post_trans(torch.Tensor(ort_output[0]).to(device)).unsqueeze(0)
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(val_ds[6]["image"][i, :, :, 70].detach().cpu(), cmap="gray")
    plt.show()
    # visualize the 3 channels label corresponding to this image
    plt.figure("label", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(val_ds[6]["label"][i, :, :, 70].detach().cpu())
    plt.show()
    # visualize the 3 channels model output corresponding to this image
    plt.figure("output", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"pth output channel {i}")
        plt.imshow(val_output[i, :, :, 70].detach().cpu())
    plt.show()
    plt.figure("output", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"onnx output channel {i}")
        plt.imshow(ort_output[0, i, :, :, 70].detach().cpu())
    plt.show()

"""## Cleanup data directory

Remove directory if a temporary was used.
"""

if directory is None:
    shutil.rmtree(root_dir)