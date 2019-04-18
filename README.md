Speaker Separation via Time-and-Frequency Cross-domain Joint Embedding
===

This is a Tensorflow implementation of speaker indepentent source separation describe in [Improved Speech Separation with Time-and-Frequency Cross-domain Joint Embedding and Clustering](https://arxiv.org/abs/1904.07845). 



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
`python main.py -m test -c path_to_model's_config`


#### Configuration
A detailed description of all configurable parameters can be found in `json/config.json`

#### Optional command-line arguments:
Argument | Valid Inputs | Default | Description
-------- | ---- | ------- | -----
mode | train/test | training |
config | string | config.json | Path to JSON-formatted config file
ckpt | string | None | Path to model's checkpoint. If not specfied, will automatically load the latest checkpoint.


Dataset
-----
Wall Street Journal with 2-speaker mixture (WSJ0-2mix)

1. [Download here](http://www.merl.com/demos/deep-clustering/)
2. Extract to `wsj0-mix/`

Citation
-----


