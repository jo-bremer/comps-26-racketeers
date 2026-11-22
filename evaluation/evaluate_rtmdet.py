import pandas as pd
import numpy as np

infer_df = pd.read_csv("RTMDet_Implementation/rtmdet_infer.csv")
gt_df = pd.read_csv("RTMDet_Implementation/rtmdet_gt.csv")

# Match predictions with annotated frames
df = gt_df.merge(
    infer_df,
    on="Frame",
    how="left",
    suffixes=("_gt", "_pred")
)

# Calculate distance between actual and predicted ball positions
df["Error"] = np.sqrt(
    (df["X_gt"] - df["X_pred"]) ** 2 +
    (df["Y_gt"] - df["Y_pred"]) ** 2
)

radius = 7

correct = (df["Error"] <= radius).sum()
predicted = df["X_pred"].notna().sum()
actual = len(df)

precision = correct / predicted if predicted else 0
recall = correct / actual if actual else 0
mde = df["Error"].mean()

print("Annotated frames:", actual)
print("Frames with predictions:", predicted)
print("Correct predictions:", correct)
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("MDE:", round(mde, 2), "pixels")