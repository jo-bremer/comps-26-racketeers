import pandas
import os
from tqdm import tqdm
import sys

def process_label_csv_for_tracknet(csv_path, output_path, img_format="jpg"):
    df = pandas.read_csv(csv_path)
    
    df = df.rename(columns={"Visibility":"visibility", "X":"x-coordinate", "Y":"y-coordinate", "Frame":"file name"})
    df["file name"] = df["file name"].apply(lambda x : f"{x:04d}.{img_format}")
    
    df.insert(0, "status", 0)
    df.to_csv(output_path, index=False)

def process_annotations_for_tracknet(path, videos_path):
    for folder in tqdm(os.scandir(path)):
        if folder.is_dir():
            for file in tqdm(os.scandir(folder.path + "/000")):
                process_label_csv_for_tracknet(file.path, videos_path+f"/_{folder.name}_000.mp4/Label.csv")
                
def main():
    assert(len(sys.argv) == 3)
    
    process_annotations_for_tracknet(sys.argv[1], sys.argv[2])
    

if __name__ == "__main__":
    main()