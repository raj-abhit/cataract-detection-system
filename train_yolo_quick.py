from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8n.pt")
    model.train(
        data=r"c:\Users\og\Desktop\praduem projectN\cataract.yolov12_ready\data.yaml",
        epochs=5,
        imgsz=512,
        batch=16,
        device=0,
        workers=0,
        project=r"c:\Users\og\Desktop\praduem projectN\runs",
        name="cataract_yolo_fast_gpu",
        exist_ok=True,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
