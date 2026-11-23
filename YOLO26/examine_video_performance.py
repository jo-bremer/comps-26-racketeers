from ultralytics import YOLO
from pathlib import Path

'''
You can edit this file to see the performance of a fine-tuned model on any of the videos in the
RacketVision dataset. However, it would make the most sense to only run this on videos in the
RacketVision testing split, which can be found in test.json.
'''
def main():
    
    PROJECT_DIR = Path(__file__).resolve().parent

    '''
    Specifies the path for the best model from a train() call. Change yolo26_run1-15 to yolo26_run1
    if you are testing this after your first time running pre_processing_and_training.py. Also
    change final_best_YOLO.pt to whatever you want to call your best model weights or simply just
    best.pt which is the default output by the train() function.
    '''
    best_path = (
        PROJECT_DIR / "runs" / "tennis"
        / "yolo26_run1-15" / "weights" / "final_best_YOLO.pt"
    )

    model = YOLO(str(best_path))
    
    '''
    Change the match number in the path below to get the annotated video for another clip.
    '''
    match_video = Path.home() / "data/RacketVision/tennis/videos/match10_000.mp4"
    
    # Shorthand to run the predict function for specified video.
    model(str(match_video), save=True, conf=0.4, classes = [0])
if __name__ == "__main__":
    main()
