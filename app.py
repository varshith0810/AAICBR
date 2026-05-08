import gradio as gr
from PIL import Image

from src.infer import load_predictor

predictor = None


def classify(image: Image.Image):
    global predictor
    if predictor is None:
        predictor = load_predictor()

    preds = predictor.predict(image, top_k=5)
    top = preds[0]
    lines = [f"Predicted Breed: {top['breed']}", "", "Top-5 confidence scores:"]
    for item in preds:
        lines.append(f"- {item['breed']}: {item['confidence']*100:.2f}%")
    return "\n".join(lines)


with gr.Blocks(title="Indian Cattle & Buffalo Breed Recognition") as demo:
    gr.Markdown("## AI-Assisted Breed Recognition for Indian Cattle and Buffaloes")
    gr.Markdown("Upload an animal image to identify its breed (software-only model; no hardware dependencies).")

    with gr.Row():
        image = gr.Image(type="pil", label="Upload Animal Image")
        output = gr.Textbox(label="Recognition Output", lines=10)

    button = gr.Button("Recognize Breed")
    button.click(classify, inputs=image, outputs=output)

if __name__ == "__main__":
    demo.launch()
