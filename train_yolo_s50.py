from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8s.pt")
    model.train(
        data=r"c:\Users\og\Desktop\praduem projectN\cataract.yolov12_ready\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        project=r"c:\Users\og\Desktop\praduem projectN\runs",
        name="cataract_yolo_s50",
        exist_ok=True,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
