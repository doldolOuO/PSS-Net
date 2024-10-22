# PSS-Net
Code for "A Progressive Shape Supplement Network with Plug-and-Play Cross-Modal Graph Attention for Point Cloud Completion".

The core code of PSS-Net is coming soon.

### Dataset
First, please download the [ShapeNetViPC-Dataset](https://pan.baidu.com/s/1NJKPiOsfRsDfYDU_5MH28A) (143GB, code: **ar8l**). Then run ``cat ShapeNetViPC-Dataset.tar.gz* | tar zx``, you will get ``ShapeNetViPC-Dataset`` contains three folders: ``ShapeNetViPC-Partial``, ``ShapeNetViPC-GT`` and ``ShapeNetViPC-View``. 

For each object, the dataset includes partial point clouds (``ShapeNetViPC-Patial``), complete point clouds (``ShapeNetViPC-GT``) and corresponding images (``ShapeNetViPC-View``) from 24 different views. You can find the detail of 24 cameras view in ``/ShapeNetViPC-View/category/object_name/rendering/rendering_metadata.txt``.

Use the code in  ``dataloader.py`` to load the dataset. 

## Training
The file config.py contains the configuration for all the training parameters.

To train the models in the paper, run this command:

```train
python train.py 
```

## Evaluation

To evaluate the models (select the specific category in config.py):

```eval
python eval.py 
```

## Acknowledgements
Some of the code of this repo is borrowed from:

- [XMFNet](https://github.com/diegovalsesia/XMFnet)

- [ChamferDistance](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)

- [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch)

## License
This project is open sourced under MIT license.
