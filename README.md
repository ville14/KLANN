# Koopman-Linearised Audio Neural Network (KLANN)

## Setup

Install the requirements.
```
python3 -m pip install -r requirements.txt
```

## Dataset
The used dataset is the one provided in [gcn-tfilm](https://github.com/mcomunita/gcn-tfilm). The dataset can be downloaded from [Zenodo](https://zenodo.org/record/7271558#.Y2I_6OzP0-R). After downloading, replace the ```data``` folder. It should have 3 subfolders: ```test```, ```train```, and ```val``` with each folder containing the input and target files. Audio should be mono.

## Training

To train, run ```python train.py```. You can change the hyperparameters in ```train.py``` to train the same models as in the paper. However, training can take a lot of time. The training results will be saved in ```results``` folder.

## Evaluate

To evaluate a trained model, run ```python eval.py```. Change the 'dir' parameter to match the name of the folder in ```results```.

## Credits
[https://github.com/mcomunita/gcn-tfilm](https://github.com/mcomunita/gcn-tfilm)

[https://github.com/boris-kuz/differentiable_iir_filters](https://github.com/boris-kuz/differentiable_iir_filters)

## Citation
If you use any of this code in your work, please consider citing us.
```    

```
