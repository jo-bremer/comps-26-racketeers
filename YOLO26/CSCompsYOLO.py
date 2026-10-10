from ultralytics import YOLO


def main():
    model = YOLO("yolo26x-seg.pt")
    model("match1_000.mp4", save=True, conf=0.4)
    
if __name__ == "__main__":
    main()