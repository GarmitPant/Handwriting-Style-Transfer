import argparse
import json
from typing import Tuple, List

import cv2
import editdistance
from path import Path
import os
from cairosvg import svg2png
import cv2
import numpy as np
import torch

from tools.cnnPrediction import get_probability_single
from tools.htr_recog import read_text
from tools.synthesis import Hand, LINES

def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='./data/recognition/line.png')
    parser.add_argument('--style', help='Target writer style 1-10.', type=int, default=1)
    parser.add_argument('--bias', help='Bias for the handwriting generation RNN.', type=float, default=0.75)

    return parser.parse_args()

def main():
    """Main function."""
    # print("YES")
    args = parse_args()
    input_text = read_text(args.img_file)
    # print("YES")
    # basename_without_ext = os.path.splitext(os.path.basename(args.img_file))[0]
    # input_text = 
    hand = Hand()
    lines = [input_text]
    biases = [args.bias for i in lines]
    styles = [args.style for i in lines]
    hand.write(
        filename='./output/usage_demo.svg',
        lines=lines,
        biases=biases,
        styles=styles
    )
    # print("YES")
    svg2png(file_obj=open("./output/usage_demo.svg", "rb"), write_to="./output/usage_demo.png")
    im = cv2.imread("./output/usage_demo.png", cv2.IMREAD_UNCHANGED)
    gray = np.array(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    cropped_image = gray[40:80, 350:700]
    new_image = np.array(cv2.resize(cropped_image, (400,100), interpolation = cv2.INTER_AREA))
    # print(new_image.shape)
    new_image = torch.Tensor(new_image).unsqueeze(0)
    # new_image = np.reshape(new_image, (1,100,400))

    print("The tensor for writer similarity is: ", get_probability_single(new_image))

if __name__ == '__main__':
  main()

