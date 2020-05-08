Implementation of Speech Separation Model 
===

This is a Tensorflow implementation of Speaker Indepentent Source Separation.

Models we have implemented are (1) [TasNet](https://ieeexplore.ieee.org/document/8707065) and (2) [Cross Domain Joint Embedding and Clustering Network](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/2181.html).

Also, we have implemented an alternative to solve the label ambiguity problem, described in [Interrupted and Cascaded PIT](https://ieeexplore.ieee.org/document/9053697).


Environments
-----
You can run on TensorFlow 2 !!! 
However, instead of executing eagerly, we build the graph first as done in TF v1.

Results
-----


| No. | Model               | Label Assignment     | SDRi (Validation) | SDRi (Test) |
| --- |--------             | ----------------     | :------------------: | :-----:        |
| (1) | Tasnet              | PIT                  | 16.2 dB              | 15.8 dB        |
| (2) | Cross-Domain        | PIT                  | 17.1 dB              | 16.9 dB        |
| (3) | TasNet              | Fixed Assign (L=100) | 17.3 dB              | 16.9 dB        |
| (4) | TasNet              | Fixed Assign (L=80)  | 17.7 dB              | 17.4 dB        |
| (5) | TasNet Init from (4)| PIT                  | 18.0 dB              | 17.7 dB        |


Usage
-----

#### Training:
`python main.py -m train -c json/config.json`
#### Testing:
`python main.py -m test -c models/name/config.json -ckpt chosen_checkpoint`


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

Reference
-----
If you find this repo interesting, you can refer to more details in the following papers. 


[1] Yang, G., Tuan, C., Lee, H., Lee, L. (2019) Improved Speech Separation with Time-and-Frequency Cross-Domain Joint Embedding and Clustering. Proc. Interspeech 2019, 1363-1367, DOI: 10.21437/Interspeech.2019-2181. [Link to paper](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/2181.html)

[2] G. Yang, S. Wu, Y. Mao, H. Lee and L. Lee, "Interrupted and Cascaded Permutation Invariant Training for Speech Separation," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 6369-6373, doi: 10.1109/ICASSP40776.2020.9053697. [Link to paper](https://ieeexplore.ieee.org/document/9053697)


