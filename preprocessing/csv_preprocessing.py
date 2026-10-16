import pandas

def process_label_csv_for_tracknet(csv_path, output_path, img_format="jpg"):
    df = pandas.read_csv(csv_path)
    
    df = df.rename(columns={"Visibility":"visibility", "X":"x-coordinate", "Y":"y-coordinate", "Frame":"file name"})
    df["file name"] = df["file name"].apply(lambda x : f"{x:04d}.{img_format}")
    
    df.add_column("status"), fill with 0s
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    process_label_csv_for_tracknet("./000_ball.csv", "./test.csv")