import annotations.annotation_visualizer as vis

ANN_PATH = "<PATH OF ANNOTATIONS CSV>"
VID_PATH = "<PATH OF UNANNOTATED VIDEO>"

def main():
    vis.visualize_ball_positions(ANN_PATH, VID_PATH, "annotated.avi")


if __name__ == "__main__":
    main()