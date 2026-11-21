import sys
from tracknet.tracknet_evaluation import evaluate_tracknet

def main():
    assert(len(sys.argv) == 3)
    evaluate_tracknet(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()