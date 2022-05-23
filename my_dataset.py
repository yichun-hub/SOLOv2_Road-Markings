from .coco import CocoDataset
from .registry import DATASETS

@DATASETS.register_module
class MyDataset(CocoDataset):
    CLASSES = ['straight arrow', 'left arrow', 
    'right arrow', 'straight left arrow', 'straight right arrow', 
    'pedestrian crossing', 'special lane']
