\# Stereo Vision Pipeline (Python)



High-performance stereo vision pipeline for depth estimation using OpenCV.



\## Features



\- Stereo matching using SGBM and BM algorithms

\- Disparity map computation

\- Depth estimation using camera calibration parameters

\- Support for Middlebury dataset

\- Visualization of disparity and depth maps

\- Batch processing of multiple scenes



\## Technologies



\- Python

\- OpenCV

\- NumPy



\## How it works



1\. Load stereo images (left/right)

2\. Compute disparity map using SGBM/BM

3\. Convert disparity to depth:

&nbsp;  

&nbsp;  Z = (f \* B) / (d + doffs)



4\. Visualize results



\## Example Usage



```bash

python src/stereo\_pipeline.py

