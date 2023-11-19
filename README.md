# fingerMinutiaeExtaction
skeleton based fingerprint minutiae extraction with OpenCv C++. 
## Overview

This program is using Latest OpenCV and performs following operations sequentially:

   - the image normalization,
   - the image orientation calculation
   - a Gabor filter based image enhancement
   - a cropping to only get the fingerprints
   - Skeletonization
   - Skeletonization Based Finger minutiae extraction

## Installation
Install OpenCV on your machine. Then clone the repo:
```bash
git clone git@github.com:osmancangudul/fingerMinutiaeExtaction.git
cd fingerMinutiaeExtaction
```

First Set the CMake folders and update the makefile for your system Than Build the cpp Code.
```bash
cmake .
make
```
## Running
For running the application you need to provide a single fingerPrint image.

```bash
./fingerMinutiaeExtaction osmi.bmp 
```

## License

This project is distributed under the Apache License 2.0.

It features code from [`edmBernard/pybind11_opencv_numpy`](https://github.com/edmBernard/pybind11_opencv_numpy)
published under the same license.
