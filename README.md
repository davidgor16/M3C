# M3C: MATRIX-STRUCTURED HIERARCHICAL CONVOLUTIONAL MODELING FOR PRONUNCIATION ASSESSMENT AND MISPRONUNCIATION DETECTION

This repository is the implementation of the paper, [**MATRIX-STRUCTURED HIERARCHICAL CONVOLUTIONAL MODELING FOR PRONUNCIATION ASSESSMENT AND MISPRONUNCIATION DETECTION**](https://ieeexplore.ieee.org/document/10095733/) (Submitted to ICASSP 2026).

> Our code is based on the open source, [https://github.com/YuanGongND/gopt](https://github.com/YuanGongND/gopt) (Gong et al, 2022).

## Dataset

An open source dataset, SpeechOcean762 (licenced with CC BY 4.0) is used. You can download it from [https://www.openslr.org/101](https://www.openslr.org/101).

## Requirements

In the repository you can find the DockerFile with the necessary requirements to run the code.
You can build the docker with the following command on your virtual environment

- `docker build --rm -t M3C_docker -f Dockerfile .`

## Data

Data folder is an empty folder. In order download the data needed to train the model, you will need to access the following [Google Drive link](https://drive.google.com/drive/folders/1a5HZ6rCQVUpEN_7xnw2HgtfV8plho7Am?usp=sharing).

## Training and Evaluation (M3C)
This bash script will run each model 5 times with ([0, 1, 2, 3, 4]).
- `cd src`
- `bash run_M3C.sh`

Note that every run does not produce the same results due to the random elements.

The reported results in the paper are the averages of the final epoch results for five different seeds.
