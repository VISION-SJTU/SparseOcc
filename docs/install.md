# Step-by-step installation instructions

SparseOcc is developed based on the official OccFormer and OpenOccupancy codebase and the installation follows similar steps.

**1. Create a conda virtual environment and activate**

python 3.8 may not be supported.
```shell
conda create -n sparseocc python=3.7 -y
conda activate sparseocc
```

**2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
We select this pytorch version because mmdet3d 0.17.1 do not supports pytorch >= 1.11 and our cuda version is 11.3 .

**3. Install mmcv, mmdet, and mmseg**
```shell
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**4. Install mmdet3d 0.17.1**

Compared with the offical version, the mmdetection3d folder in this repo further includes operations like bev-pooling. 

```shell
cd mmdetection3d
pip install -r requirements/runtime.txt
python setup.py install
cd ..
```

**5. Build dependencies**
```shell
cd SparseOcc
export PYTHONPATH=“.”
python install -v -e .
```

**d. Install other dependencies, like timm, einops, torchmetrics, etc.**

Please change the spconv version according to your cuda version.
```shell
pip install -r docs/requirements.txt
```