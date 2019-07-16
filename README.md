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
If you find this repo helpful, please kindly cite my paper. 

```
@article{yang2019improved,
  title={Improved Speech Separation with Time-and-Frequency Cross-domain Joint Embedding and Clustering},
  author={Yang, Gene-Ping and Tuan, Chao-I and Lee, Hung-Yi and Lee, Lin-shan},
  journal={arXiv preprint arXiv:1904.07845},
  year={2019}
}
```

