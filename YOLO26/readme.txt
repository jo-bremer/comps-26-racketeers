1. RacketVision dataset

Get the tennis portion of RacketVision separately. The dataset is not included
with this project. Put it in ~/data/RacketVision/tennis/ (~ is your home folder).
Keep these three folders together:

    videos/   Tennis videos in MP4 format.
    all/      CSV files that say when and where the ball is visible.
    info/     Lists of videos used for training, validation, and testing.

2. Pretrained YOLO26x

Get the COCO-pretrained YOLO26x weights from ultralytics. The weights are not included
with this project. Put the file named yolo26x.pt next to runModel.py.

Run the commands below from the YOLO26 folder. The scripts are set up to use
an NVIDIA GPU. In a new Python environment, install the packages first:

    python3 -m pip install -r requirements.txt pandas

3. runModel.py
Try the pretrained model
    mkdir -p output_csv
    python3 runModel.py
This script looks for sports balls in each video. It filters out detections
that stay in the same spot. It writes one CSV per video in output_csv directory.

Each row means a frame. It gives bounding box and confidence level for each 
detection. You can skip this step if you want to go straight to training.

4. pre_processing_and_training.py
Prepare images and fine-tune the model

Before your first run:
    mkdir -p images/{train,val,test} labels/{train,val,test}
Open videos.yaml. Set "path" to the full path of your YOLO26 folder.
Under "names", set class 0 to "ball" so evaluate do the correct evaluation.

Look in labels/ for files such as match1.csv. If the match CSVs are missing,
open pre_processing_and_training.py. In main(), remove the leading # from
the line that calls copy_ground_truth_csvs(...). Then run:
    python3 pre_processing_and_training.py
The script saves labeled video frames as JPG. For each visible ball,
it writes a label with a 20x20 pixel box. It leaves the label file empty
when the ball is hidden. It then fine-tunes YOLO26x on the training images.
Fine-tuning means starting with the pretrained weights and updating them
using the tennis images and labels.

Look at folder under runs/tennis/. The file weights/best.pt
has the weights from the epoch with the best result evaluated on val split.

5. evaluate.py
Test your fine-tuned model

The test images are ready on the project server. Run this from the YOLO26 folder:
    python3 evaluate.py --weights runs/tennis/yolo26_run1/weights/best.pt --imgsz 1280

Use your own best.pt path after --weights. Choose corresponding image size after --imgsz.
To see your scores, open metrics.csv in the new folder under runs/evaluation/.