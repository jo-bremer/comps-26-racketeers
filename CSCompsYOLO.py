from ultralytics import YOLO


def main():
    model = YOLO("yolo26x-seg.pt")
    
    
if __name__ == "__main__":
    main()