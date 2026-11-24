import pandas
import os
from tqdm import tqdm
import sys
import json

def process_infer_csv_from_tracknet(csv_path):
    """Processes tracknet data back to original RacketVision format"""
    
    df = pandas.read_csv(csv_path, engine="python", skipfooter=2)
    
    df = df.rename(columns={"detected" : "Visibility", "frame_number" : "Frame", "x":"X", "y":"Y", "conf" : "Confidence"})
    
    return df