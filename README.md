# SOLOv2_Road-Markings
  ## 1. Converts the format of VGG Image Annotator (VIA) to the COCO format.
  Classes:straight arrow, left arrow, right arrow, straight left arrow, straight right arrow,  pedestrian crossing, special lane
  ## 2. SOLOv2 environment
  * Operating System: Ubuntu 20.04.4
  * GPU: NVIDIA GeForce RTX3090
  * CUDA 11.1
  * pytorch 1.8.0
  * torchvision 0.9.0
  * python 3.7.13

  ## 3. Create my_dataset.py
  Create a python file for the classes of the custom dataset in `mmdet/datasets`
  ```python
  from .coco import CocoDataset
  from .registry import DATASETS

  @DATASETS.register_module
  class MyDataset(CocoDataset):
      CLASSES = ['straight arrow', 'left arrow', 
      'right arrow', 'straight left arrow', 'straight right arrow', 
      'pedestrian crossing', 'special lane']
  ```
  Add the dataset in the `mmdet/datasets/__init__.py`
  
  ## 4. Modify solov2_r101_fpn_8gpu_3x.py
  Backbone: ResNet101+FPN
  * Download pretrained model: https://github.com/pytorch/vision/blob/d585f86d94f07a3bc083e48c6534d93a409cbcb2/torchvision/models/resnet.py#L312
  * Modify classes number `num_classes`: classes number + background
  * Modify number of grids `num_grids`:[80, 72, 64, 48, 32] (if the image size is larger)
  * Modify the dataset settings `dataset_type` `data_root`
  * Modify train/val/test images and annotations path `ann_file` `img_prefix`
  * Modify runtime settings `total_epochs` `work_dir` (the path of the trained weights)

  ## 5. Training
      python tools/train.py configs/solov2/solov2_r101_fpn_8gpu_3x.py
      
  ## 6. Evaluation
      python tools/test_ins.py configs/solov2/solov2_r101_fpn_8gpu_3x.py weights/homo_model_2/epoch_100.pth --show --out results_solo.pkl --eval segm
      
  ## 7. Visulization
  The `class_names`should be modified.
      python tools/test_ins_vis.py configs/solov2/solov2_r101_fpn_8gpu_3x.py weights/homo_model_2/latest.pth --show --save_dir  work_dirs/val_homo_2data
