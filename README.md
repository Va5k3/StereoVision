# Stereo Vision 

## About
Implementation of **stereo vision** in Python using image pairs from the **Middlebury dataset**.  
The main idea is: given a **left** and **right** photo of the same scene, we compute a **disparity map** and then estimate a **depth map**.

I used OpenCV’s built-in stereo matchers:
- **StereoBM** (Block Matching)
- **StereoSGBM** (Semi-Global Block Matching)

This project is mainly for learning + experimenting with parameters (disparity range, block size, etc.) and seeing how they affect the result.

---

## What it does
- Takes a rectified stereo pair (left/right)
- Computes a **disparity map**
- Converts disparity to a **depth map** (approximate)
- Saves output images (and/or displays them)

---

## Requirements
- Python 3.x
- **OpenCV** (`opencv-python`)
- **NumPy**

Example install:
```bash
pip install opencv-python numpy
```

> If you have a `requirements.txt`, use:
```bash
pip install -r requirements.txt
```

---

## Dataset (Middlebury)
This project uses stereo pairs from the **Middlebury dataset**.

### Download
You need to download the dataset **separately** (it’s not included in this repo).

- Dataset version: **Middlebury 2021**


---

## How to run
Right now you said the command is:

```bash
python main
```


## Methods (quick explanation)
### StereoBM (Block Matching)
- Compares small windows (blocks) between the left and right image
- Faster and simpler
- Usually noisier / more “blocky” results

### StereoSGBM (Semi-Global Block Matching)
- Uses a smoother optimization idea compared to BM
- Often produces cleaner disparity maps
- Slower but typically better quality

---

## Depth map (from disparity)
Depth is inversely related to disparity. A common approximate relationship is:

**depth ≈ (focal_length * baseline) / disparity**

If calibration values aren’t available, the depth map is often “relative” (useful for visualization, but not exact meters).

---

## Results 
<img width="1911" height="1080" alt="Screenshot 2026-03-17 190208" src="https://github.com/user-attachments/assets/0f23cec3-f4a2-46d4-8fc3-11fc51ba9cd7" />
<img width="461" height="296" alt="image" src="https://github.com/user-attachments/assets/b55b0514-4978-4b9d-bd81-13e6cbeb206b" />


---

## C++ implementation

A native OpenCV build lives under **`cpp/`** (CMake + optional vcpkg manifest). It mirrors the Python pipeline: same calibration and stereo matchers, disparity → depth, visualization and optional export.

See **[cpp/README.md](cpp/README.md)** for configure, build, and run instructions.

---

## Tech stack
- Python, OpenCV, NumPy
- C++20, OpenCV 4 (see `cpp/`)
