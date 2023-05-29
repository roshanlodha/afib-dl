# AFib Recurrence Prediction using Transfer Learning (VGG16 and ImageNet)

This project utilizes transfer learning techniques with the VGG16 model pre-trained on the ImageNet dataset to predict the recurrence of atrial fibrillation (AFib) based on segmented image files. The segmentation process specifically focuses on the pulmonary veins and heart chambers.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Atrial fibrillation (AFib) is a common heart condition that causes irregular and often rapid heart rate, leading to various health risks. Early detection and prediction of AFib recurrence can significantly aid in proactive medical interventions. This project aims to leverage the power of transfer learning and deep neural networks to predict AFib recurrence using segmented image files that focus on the pulmonary veins and heart chambers.

## Dataset
The dataset used in this project consists of segmented image files capturing the pulmonary veins and heart chambers. These images are collected from various medical imaging sources and have been manually labeled for AFib recurrence. The dataset is not included in this repository due to privacy and legal restrictions. However, instructions for acquiring and preparing the dataset will be provided in the [Usage](#usage) section.

## Project Structure
The project follows the following directory structure:
├── data/
│ ├── train/
│ ├── validation/
│ └── test/
├── models/
│ └── vgg16.h5
├── src/
│ ├── data_processing.py
│ ├── model_training.py
│ └── prediction.py
├── README.md
└── requirements.txt

- **data/**: This directory contains the training, validation, and testing sets. Segmented image files should be placed in their respective folders.
- **models/**: This directory stores the pre-trained VGG16 model file.
- **src/**: This directory contains the source code files for data processing, model training, and prediction.
- **README.md**: The file you are currently reading.
- **requirements.txt**: A file listing the dependencies required to run the project.
