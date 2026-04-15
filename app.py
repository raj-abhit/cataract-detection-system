from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(page_title="Cataract Detection", layout="wide")

MODEL_PATH = Path("runs/cataract_yolo_s50/weights/best.pt")
EXAMPLES_DIR = Path("examples")
MAIN_MODEL_METRICS = {
    "validation": {"precision": 0.820, "recall": 0.826, "map50": 0.842, "map50_95": 0.534},
    "test": {"precision": 0.823, "recall": 0.834, "map50": 0.863, "map50_95": 0.543},
    "test_classwise": {
        "Cataract": {"precision": 0.777, "recall": 0.772, "map50": 0.852, "map50_95": 0.500},
        "Normal": {"precision": 0.868, "recall": 0.896, "map50": 0.873, "map50_95": 0.586},
    },
}


@st.cache_resource
def load_model() -> YOLO:
    return YOLO(str(MODEL_PATH))


def predict_and_render(model: YOLO, image: Image.Image):
    results = model.predict(source=np.array(image), conf=0.25, verbose=False)
    result = results[0]
    plotted = result.plot()
    plotted = Image.fromarray(plotted[:, :, ::-1])
    return result, plotted


def main() -> None:
    st.title("Cataract Detection UI")
    st.write("Test with a sample image or upload your own eye image.")
    st.caption("Model: YOLOv8s (main model: cataract_yolo_s50)")

    with st.expander("Main Model Evaluation Scores", expanded=True):
        val = MAIN_MODEL_METRICS["validation"]
        test = MAIN_MODEL_METRICS["test"]
        st.markdown("**Validation**")
        st.write(
            f"- Precision: {val['precision']:.3f} | Recall: {val['recall']:.3f} | "
            f"mAP50: {val['map50']:.3f} | mAP50-95: {val['map50_95']:.3f}"
        )
        st.markdown("**Test**")
        st.write(
            f"- Precision: {test['precision']:.3f} | Recall: {test['recall']:.3f} | "
            f"mAP50: {test['map50']:.3f} | mAP50-95: {test['map50_95']:.3f}"
        )
        st.markdown("**Test Class-wise**")
        for cls_name, m in MAIN_MODEL_METRICS["test_classwise"].items():
            st.write(
                f"- {cls_name}: P={m['precision']:.3f}, R={m['recall']:.3f}, "
                f"mAP50={m['map50']:.3f}, mAP50-95={m['map50_95']:.3f}"
            )

    if not MODEL_PATH.exists():
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

    model = load_model()

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Input")
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=False,
        )

        sample_files = sorted(EXAMPLES_DIR.glob("*"))
        sample_name = st.selectbox(
            "Or choose a sample image",
            ["None"] + [p.name for p in sample_files],
            index=0,
        )

    input_image = None
    source_label = None
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        source_label = f"Uploaded: {uploaded_file.name}"
    elif sample_name != "None":
        sample_path = EXAMPLES_DIR / sample_name
        input_image = Image.open(sample_path).convert("RGB")
        source_label = f"Sample: {sample_name}"

    with right:
        st.subheader("Result")
        if input_image is None:
            st.info("Upload an image or pick one sample to run inference.")
            return

        st.caption(source_label)
        result, plotted_image = predict_and_render(model, input_image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(input_image, caption="Input image", use_container_width=True)
        with col2:
            st.image(plotted_image, caption="Prediction", use_container_width=True)

        if result.boxes is None or len(result.boxes) == 0:
            st.warning("No object detected.")
            return

        names = result.names
        top_idx = int(result.boxes.conf.argmax().item())
        top_cls_id = int(result.boxes.cls[top_idx].item())
        top_conf = float(result.boxes.conf[top_idx].item())
        st.success(f"Top prediction: {names[top_cls_id]} ({top_conf:.2%})")

        st.markdown("**Detected boxes**")
        for cls_id, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            st.write(f"- {names[int(cls_id)]}: {float(conf):.2%}")


if __name__ == "__main__":
    main()
