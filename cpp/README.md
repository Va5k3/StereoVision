# C++ implementation

Same pipeline as the Python project: **StereoBM / StereoSGBM**, Middlebury-style scene folders (`calib.txt` + stereo pair), disparity and depth visualization.

## Requirements

- CMake 3.20+
- C++20 compiler (MSVC, Clang, GCC)
- OpenCV with `core`, `imgproc`, `imgcodecs`, `calib3d`, `highgui`

## Build (vcpkg)

From the **repository root** (where `main.py` and `all/` live):

```bash
cd cpp
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --config Release
```

On Windows, replace `/path/to/vcpkg` with your vcpkg clone (or set `VCPKG_ROOT` and pass `-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake`).

Dependencies are listed in `vcpkg.json` (OpenCV 4). Install the triplet you use, for example:

```bash
vcpkg install opencv4:x64-windows
```

## Run

Paths are **relative to the current working directory**:

```bash
# from repository root, after build (MSVC multi-config):
cpp/build/Release/stereovision.exe all/data/curule1
# single-config generator (Ninja/Make):
cpp/build/stereovision all/data/curule1
```

Or set `STEREOVISION_DATASET` to the scene directory instead of passing an argument.

Outputs go to `./stereo_output/` by default (PNG + YAML), same idea as the Python tool.

## Layout

| Path | Role |
|------|------|
| `CMakeLists.txt` | Build definition |
| `vcpkg.json` | Manifest dependencies |
| `include/stereovision/` | Public headers |
| `src/` | Implementation |
