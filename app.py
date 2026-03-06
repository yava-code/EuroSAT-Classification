"""
EuroSAT Classifier — Gradio demo for Hugging Face Spaces.
Upload a satellite image → get land-use class predictions.
"""

import torch
import gradio as gr
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image

from model import SimpleNet, CLASS_NAMES


# ── Load model ────────────────────────────────────────────────────────

def load_model():
    """Download weights from HF Hub and load into SimpleNet."""
    # TODO: replace with your actual HF repo id after upload
    weights_path = hf_hub_download(
        repo_id="yava-code/eurosat-simplenet",
        filename="simple_net_v1.pth",
    )
    model = SimpleNet(num_classes=10)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


model = load_model()

preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Inference ─────────────────────────────────────────────────────────

def predict(image: Image.Image) -> dict[str, float]:
    """Return class probabilities for a satellite image."""
    if image is None:
        return {}

    tensor = preprocess(image).unsqueeze(0)  # [1, 3, 64, 64]

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


# ── Gradio UI ─────────────────────────────────────────────────────────

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Satellite Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="🛰️ EuroSAT Land-Use Classifier",
    description=(
        "Upload a Sentinel-2 satellite image to classify its land-use type. "
        "Custom CNN (SimpleNet, ~850K params) trained from scratch on EuroSAT."
    ),
    examples=[],  # add example images if you want
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
