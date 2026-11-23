import pandas
import os
from tqdm import tqdm
import sys
import json

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

def combine_csvs(csv_1, csv_2):
    df_1 = pandas.read_csv(csv_1)
    df_2 = pandas.read_csv(csv_2)
    df_con = pandas.concat([df_1, df_2], )
    df_con.to_csv("out.csv", index=False)
    
def get_split(csv_in, json_split):
    with open(json_split, "r") as file:
        raw_data = json.load(file)
    data = ""
    for match, _ in raw_data:
        data += f"{match}_|"
    data = data[:-1]
    
    df_in = pandas.read_csv(csv_in)
    
    df_out = df_in[df_in["path_prev"].str.contains(data)]
    df_out.to_csv("split_out.csv")
    
def main():
    assert(len(sys.argv) == 4)
    
    match(sys.argv[1]):
        case "split":
            get_split(sys.argv[2], sys.argv[3])
        case "combine":
            combine_csvs(sys.argv[2], sys.argv[3])
        case "tracknet":
            process_annotations_for_tracknet(sys.argv[2], sys.argv[3])
        case _:
            print("split <csv_in> <json_in>\ncombine <csv_1> <csv_2>\ntracknet <csv_in> <csv_out>")
    

if __name__ == "__main__":
    main()