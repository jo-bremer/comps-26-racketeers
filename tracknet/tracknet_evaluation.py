import pandas as pd
import os
from tqdm import tqdm
import sys
import json
from postprocessing.csv_postprocessing import process_infer_csv_from_tracknet
from evaluation.evaluation import evaluate

def evaluate_tracknet(annotations_dir, infers_dir):
    infer_subdirs = list(os.listdir(infers_dir))
    data = []
    
    for infer_dirname in infer_subdirs:
        match_id = infer_dirname.split("_")[0]
        if match_id == "":
            continue
        
        annotation_csv = f"{annotations_dir}/{match_id}/000/results.csv"
        infer_csv = f"{infers_dir}/{infer_dirname}/{infer_dirname}_data.csv" 
    
        gt_df = pd.read_csv(annotation_csv)
        infer_df = process_infer_csv_from_tracknet(infer_csv)
        
        gt_df = gt_df.head(len(infer_df))
        
        prec, rec, mde, map50 = evaluate(infer_df, gt_df)
        
        info = {"Match" : match_id, "Precision" : prec, "Recall" : rec, "MDE" : mde, "mAP@50" : map50}
        data.append(info)
    
    df = pd.DataFrame(data)
    df.to_csv("out.csv", index = False)
    
    avg_info = [{"Precision" : df["Precision"].mean(), "Recall" : df["Recall"].mean(), "MDE" : df["MDE"].mean(), "mAP@50" : df["mAP@50"].mean()}]
    pd.DataFrame(avg_info).to_csv("avg_out.csv", index=False)