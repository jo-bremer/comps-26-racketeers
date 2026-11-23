import numpy as np
import pandas as pd

def build_eval_df(infer_df, gt_df, radius=5, scale=1.0, iou50 = 7):
    df = infer_df.merge(gt_df, on="Frame", suffixes=("_pred", "_gt"))
    df["Distance Error"] = np.hypot(
        df["X_pred"] / scale - df["X_gt"],
        df["Y_pred"] / scale - df["Y_gt"],
    )
    pred_vis = df["Visibility_pred"] > 0
    gt_vis = df["Visibility_gt"] > 0
    df["Correct"] = (pred_vis & gt_vis & (df["Distance Error"] <= radius))
    df["Half Overlap"] = (pred_vis & gt_vis & (df["Distance Error"] <= iou50))
    return df

def eval_precision(df):
    n_pred = (df["Visibility_pred"] > 0).sum()
    return df["Correct"].sum() / n_pred if n_pred else 0.0

def eval_recall(df):
    n_gt = (df["Visibility_gt"] > 0).sum()
    return df["Correct"].sum() / n_gt if n_gt else 0.0

def eval_MDE(df):
    return df.loc[(df["Visibility_pred"] > 0) & (df["Visibility_gt"] > 0), "Distance Error"].mean()

def eval_mAP50(df):
    n_gt = (df["Visibility_gt"] > 0).sum()
    preds = df[df["Visibility_pred"] > 0].sort_values("Confidence_pred", ascending=False)
    if n_gt == 0 or preds.empty:
        return 0.0

    tp = preds.loc[preds["Half Overlap"] > 0, "Correct"].cumsum()
    fp = (1 - preds.loc[preds["Half Overlap"] > 0, "Correct"]).cumsum()
    
    prec = tp / (tp + fp)
    rec = tp / n_gt
    rec_prev = rec.shift(fill_value=0)
    
    return float((prec * (rec - rec_prev)).sum())

def evaluate(infer_df, gt_df):
    assert set(infer_df.columns) == set(gt_df.columns) == {"Frame", "Visibility", "X", "Y", "Confidence"}

    df = build_eval_df(infer_df, gt_df)
    prec, rec, mde, map50 = eval_precision(df), eval_recall(df), eval_MDE(df), eval_mAP50(df)
    print(f"[Precision : {prec} | Recall : {rec} | MDE : {mde} | mAP@50 : {map50}]")
    
    return prec, rec, mde, map50