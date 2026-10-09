from pandas import DataFrame, read_csv

def read_ball_position_annotations(ball_position_annotations_path : str) -> DataFrame:
    df : DataFrame = read_csv(ball_position_annotations_path)
    frames = {}
    
    for line in df.to_dict(orient="records"):
        if line["Visibility"] == 1:
            frames[line["Frame"]] = (line["X"], line["Y"])
    return frames