#! /bin/bash
set -euo pipefail

rm -rf ./isic2019/ && mkdir -p ./isic2019/

# Download and unzip
echo "Downloading ISIC 2019 training data..."
curl -SL https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip > ./isic2019/ISIC_2019_Training_Input.zip
unzip ./isic2019/ISIC_2019_Training_Input.zip -d ./isic2019

echo "Downloading ISIC 2019 ground truth..."
curl -SL https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv > ./isic2019/ISIC_2019_Training_GroundTruth_Original.csv
