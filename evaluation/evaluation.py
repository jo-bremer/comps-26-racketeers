import pandas as pd
import math

def evaluate_prediction_correctness(infer_df, gt_df, correct_prediction_radius=7, scale=0.266666667):
    infer_df["Correct"] = 0
    infer_df.loc[( \
        (infer_df["X"]/scale - gt_df.loc[gt_df["Frame"] == infer_df["Frame"], "X"])**2 +
        (infer_df["Y"]/scale - gt_df.loc[gt_df["Frame"] == infer_df["Frame"], "Y"])**2)**0.5 
                 <= correct_prediction_radius, \
    "Correct"] = 1
    
    return infer_df



def eval_precision(infer_df, gt_df):
    
    raise Exception("TODO - Evaluate Precision")
    return -1.0

def eval_recall(infer_df, gt_df):
    raise Exception("TODO - Evaluate Recall")
    return -1.0

def eval_MDE(infer_df, gt_df):
    # ASSUMING 1920x1080 - INSURE
    raise Exception("TODO - Evaluate MDE")
    return -1.0

def eval_mAP50(infer_df, gt_df):
    raise Exception("TODO - Evaluate mAP@50")
    return -1.0



def evaluate(infer_csv, gt_csv):
    infer_df = pd.read_csv(infer_csv)
    gt_df = pd.read_csv(gt_csv) #Use the interp_ball for gt!
    
    gt_df = gt_df.drop(['Confidence'], axis=1, errors='ignore')
    
    assert(set(infer_df.columns.tolist()) == set(gt_df.columns.tolist())\
        == set(["Frame", "Visibility", "X", "Y"]))
    
    evaluate_prediction_correctness(infer_df, gt_df)
    
    prec, rec, mde, map50 = \
        eval_precision(infer_df, gt_df),\
        eval_recall(infer_df, gt_df),\
        eval_MDE(infer_df, gt_df),\
        eval_mAP50(infer_df, gt_df)
    
    print(f"[Precision : {prec} | Recall : {rec} | MDE : {mde} | mAP@50 : {map50}]")

def main():
    evaluate("./evaluation/test_infer.csv", "./evaluation/test_gt.csv")

if __name__ == "__main__":
    main()