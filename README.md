# SeqSleepNet
- Huy Phan, Fernando Andreotti, Navin Cooray, Oliver Y. Chén, and Maarten De Vos. [__SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging.__](https://arxiv.org/pdf/1809.10932) _arXiv Preprint  	arXiv:1809.10932,_ 2018

These are source code and experimental setup for the __MASS database__, used in our above arXiv preprint. Although the networks have many things in common, we try to separate them and to make them work independently to ease exploring them invididually.

Currently, _SeqSleepNet_ and two baselines _E2E-ARNN_ and _Multitask E2E-ARNN_ are available (_E2E-DeepSleepNet_ baseline is still missing, we will clean it up and make it available shortly). Output of t nheetworks are also included, so that you can re-produce the results with the evaluation scripts. However, you can repeat the experiments following the steps below. 

How to run:
-------------
1. Download the databases
- __MASS database__ is available [here](https://massdb.herokuapp.com/en/). Information on how to obtain it can be found therein.
2. Data preparation
- Change directory to `./data_processing/`
- Run `main_run.m`
3. Network training and testing
- Change directory to a specific network in `./tensorflow_net/`, for example `./tensorflow_net/SeqSleepNet/`
- Run a bash script, e.g. `bash run_seq20.sh`, to repeat 20 cross-validation folds.  
_Note1:_ You may want to modify and script to make use of your computational resources, such as place a few process them on multiple GPUs. If you want to run multiple processes on a single GPU, you may want to modify the Tensorflow source code to change __GPU options__ when initializing a Tensorflow session. 
4. Evaluation
- Execute a specific evaluation Matlab script, for example `eval_seqsleepnet.m`

Environment:
-------------
- Matlab v7.3 (for data preparation)
- Python3
- Tensorflow GPU 1.3.0 (for network training and evaluation)

Contact:
-------------
Huy Phan  
Institute of Biomedical Engineering  
Department of Engineering Science  
University of Oxford  
Email: huy.phan{at}eng.ox.ac.uk

