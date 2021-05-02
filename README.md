# Visual-SLAM

## Introduction
Code for implementation of Visual SLAM using data from an RBG Camera equipped on a Autonomous Vehicle using Eight Point Algorithm for the 3D Point Clouds data.

## Overview of the Repository
In this repo, you'll find :
* `Kitti`: famous kitti dataset.
* `hartley1997.pdf`: paper of the Eight Point Algorithm for 3D Point Clouds data.
* `motion2D2D.py`: motion estimation from 2 frames in a monocular setup.
* `draw.py`: modified version of opencv DrawMatches to recover randomly generated colors

## Getting Started
1.  Clone repo: `git clone https://github.com/HusseinLezzaik/Visual-SLAM.git`
2.  Install dependencies:
    ```
    conda create -n visual-servoing python=3.7
    conda activate visual-servoing
    pip install -r requirements.txt
    ```

And you're good to go!

## Contact
* Hussein Lezzaik : hussein dot lezzaik at gmail dot com


