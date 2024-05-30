#!/bin/bash

resolution=256
while [[ $resolution -le 35000 ]]; do
    cmake-build-debug/dataset_generator dataset/${resolution}x${resolution}.ppm $resolution $resolution
    resolution=$((resolution * 2))
done

