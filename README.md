# M3C: Matrix-Structured Hierarchical Convolutional Modeling for Pronunciation Assessment and Mispronunciation Detection

This repository is the implementation of the paper, **Matrix-Structured Hierarchical Convolutional Modeling for Pronunciation Assessment and Mispronunciation Detection** (Submitted to ICASSP 2026).

> Our code is based on the open source, [https://github.com/YuanGongND/gopt](https://github.com/YuanGongND/gopt) (Gong et al, 2022).

## Citation

`@INPROCEEDINGS{m3c,
  author={Fernández-García, David and González-Ferreras, César and Cardeñoso-Payo, Valentín and Corrales-Astorgano, Mario},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Matrix-Structured Hierarchical Convolutional Modeling for Pronunciation Assessment and Mispronunciation Detection}, 
  year={2026},
  volume={},
  number={},
  pages={17567-17571},
  keywords={Jamming;Feeds;Filtering;Filters;Feedback;Circuits;Circuits and systems;Protocols;HTTP;LoRa;computer-assisted pronunciation training;automatic pronunciation assessment;mispronunciation detection;CNN},
  doi={10.1109/ICASSP55912.2026.11462089}}`

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
