# Multiple Granularity Network
Implement of paper:[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)

## Dependencies

- Python >= 3.5
- PyTorch >= 0.4.0
- torchvision
- scipy
- numpy
- scikit_learn
- tqdm
- matplotlib
- h5py

## Data

The data structure would look like:
```
data/
    bounding_box_train/
    bounding_box_test/
    query/
```
#### Market1501 
Download from [here](http://www.liangzheng.org/Project/project_reid.html)

## Train

You can specify more parameters in opt.py

```
python main.py --mode train --data_path <path/to/Market-1501-v15.09.15> 
```

## Evaluate

Use pretrained weight or your trained weight

```
python main.py --mode evaluate --data_path <path/to/Market-1501-v15.09.15> --weight <path/to/weight_name.pt> 
```
