# General Neural Combinatorial Optimization with Problem Instance Reduction

A general framework for multi-task and multi-scale neural combinatorial optimization utilizing problem reduction.

This implementation contains our improved transformer-based `MatNet-POE` model and adapted existing gnn-based `DIMES` model.

### Dependencies

- pip install tsplib95

- pip install lkh

- pytorch 1.11.0+cu113


### Pretrained models
Checkpoints for `DIMES` are provided with codes. Pretrained model for our `MatNet-POE` can be downloaded [here](https://drive.google.com/file/d/16mDY9HVzDdyFnqrL6YnrQ2lS8twscD_o/view?usp=sharing). Place the unzipped files under `ckpts` folder for evaluation.

### Datasets
Datasets with `N=20` are provided with codes. Please download other datasets [here](https://drive.google.com/file/d/17LINJtArttm8ba6VEQ4XdfGjuz-ZMl3I/view?usp=sharing) and place `val_sets` and `test_sets` under `data` folder for full evaluation and testing. Training instances are randomly generated on the fly.

Note that you can modify arguments specified in `train.py` and `test.py` respectively for customized execution.

### Training & Evaluation

Prior to training/testing, run:
```
gcc utils/base_methods.c -o libtsp.so -fPIC -shared
```
#### MatNet-POE
Run following lines for your quicks reference on TSP-20:
```
cd MatNet-POE
python train.py
python test.py
```

#### DIMES
Run following lines for your quick reference on TSP-20:
```
cd DIMES
python train.py
python test.py
```

Training results and checkpoints shall be stored in `result` folder for either model.