import annotations.annotation_visualizer as vis
import tools.preprocess_data_gauss as preprocess

ANN_PATH = "C:/Users/jmark/OneDrive/Desktop/Work/COMPS/Racketeers/data/tennis/all/match1/csv/000_ball.csv"
VID_PATH = "C:/Users/jmark/OneDrive/Desktop/Work/COMPS/Racketeers/data/tennis/videos/match1_000.mp4"


# C:/Users/jmark/OneDrive/Desktop/Work/COMPS/Racketeers/comps-26-racketeers/preprocessing/data

def main():
    vis.visualize_ball_positions(ANN_PATH, VID_PATH, "annotated.avi")


if __name__ == "__main__":
    main()
    
#COMPSF2026!=MoviePoster
#bremerj@olin312-57.cslabs.carleton.edu