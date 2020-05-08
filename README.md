Improved Speech Separation with Time-and-Frequency Cross-domain Joint Embedding and Clustering
===

This is a Tensorflow implementation of speaker indepentent source separation describe [here](https://arxiv.org/abs/1904.07845). 


Environments
-----
1. Tenforflow 1.13
2. mir_eval 0.5
3. scipy 1.2.1


Usage
-----


#### Training:
`python main.py -m train -c json/config.json`
#### Testing:
`python main.py -m test -c models/name/config.json`


#### Configuration
A detailed description of all configurable parameters can be found in `json/config.json`

#### Optional command-line arguments:
Argument | Valid Inputs | Default | Description
-------- | ---- | ------- | -----
mode | train/test | training |
config | string | config.json | Path to JSON-formatted config file
ckpt | string | None | Path to model's checkpoint. If not specfied, will automatically load the latest checkpoint.


Dataset Preprocess
-----
From SPHERE to wav : `bash convert_wsj0.sh`

Generate WSJ0-2mix (Wall Street Journal with 2-speaker mixture) or WSJ0-3mix

1. Download [official code](http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip) or use my modified version in `create_wav_2speakers.m` and `create_wav_3speakers.m`
2. Download [voicebox](https://github.com/ImperialCollegeLondon/sap-voicebox/tree/master/voicebox)
3. Steps to run octave on linux:

    (1) run `octave-cli`

    (2) load package `pkg load <pkg-name>`
    	
    (3) run `create_wav_2speakers.m` or `create_wav_3speakers.m`

Citation
-----
If you find this repo helpful, please kindly cite our paper. 


[1] Yang, G., Tuan, C., Lee, H., Lee, L. (2019) Improved Speech Separation with Time-and-Frequency Cross-Domain Joint Embedding and Clustering. Proc. Interspeech 2019, 1363-1367, DOI: 10.21437/Interspeech.2019-2181.

[2] G. Yang, S. Wu, Y. Mao, H. Lee and L. Lee, "Interrupted and Cascaded Permutation Invariant Training for Speech Separation," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 6369-6373, doi: 10.1109/ICASSP40776.2020.9053697.


