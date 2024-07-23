## SemanticKITTI

To prepare for SemanticKITTI dataset, please download the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (including color, velodyne laser data, and calibration files) and the annotations for Semantic Scene Completion from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download). Put all `.zip` files under `SparseOcc/data/SemanticKITTI` and unzip these files. Then you should get the following dataset structure:
```
SparseOcc
├── data/
│   ├── SemanticKITTI/
│   │   ├── dataset/
│   │   │   ├── sequences
│   │   │   │   ├── 00
│   │   │   │   │   ├── calib.txt
│   │   │   │   │   ├── image_2/
│   │   │   │   │   ├── image_3/
│   │   │   │   │   ├── voxels/
│   │   │   │   ├── 01
│   │   │   │   ├── 02
│   │   │   │   ├── ...
│   │   │   │   ├── 21
```

Preprocess the annotations for semantic scene completion:
```bash
python projects/mmdet3d_plugin/tools/kitti_process/semantic_kitti_preprocess.py --kitti_root data/SemanticKITTI --kitti_preprocess_root data/SemanticKITTI --data_info_path projects/mmdet3d_plugin/tools/kitti_process/semantic-kitti.yaml
```

## OpenOccupancy 
Please refer to [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md) for dataset download and preparation.
The final folder structure should be
```
SparseOcc
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── nuscenes_occ_infos_train.pkl/
│   │   ├── nuscenes_occ_infos_val.pkl/
│   ├── depth_gt/
│   ├── nuScenes-Occupancy/