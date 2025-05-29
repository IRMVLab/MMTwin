# MMTwin: New Path to Human-Robot Policy Transfer?

<img src="https://github.com/IRMVLab/MMTwin/blob/main/docs/title_page.png" />

The code and pretrained models for our paper "Novel Diffusion Models for Multimodal 3D Hand Trajectory Prediction" will be released here. Please refer to our [paper](https://arxiv.org/abs/2504.07375) and the [project page](https://irmvlab.github.io/mmtwin.github.io) for more details.

We are updating the tutorials. Your patience is appreciated :)


## TODO
- [x] Release the paper  :bowtie:
- [x] Release our self-collected CABH Benchmark for fast HTP evaluation :sunglasses:	
- [ ] Release the code and pretrained models

## Suggested Data Structure
```
HTPdata/
|-- EgoPAT3D-postproc
      |-- odometry
      |-- trajectory_repair
      |-- video_clips_hand
      |-- pointcloud_bathroomCabinet_1  # for demo
      |-- glip_feats
      |-- motion_feats
      |-- egopat_voxel_filtered
|-- CABH-benchmark
```




## Benchmark Evaluation

### EgoPAT3D

#### 0. Data Preprocessing (optional)

* Camera Egomotion Generation 

* Arm Filtering for Clean Global Context

<div style="display: flex; justify-content: space-between; width: 100%;">
  <img src="./docs/raw_image.png" style="height: 120px; object-fit: contain; flex: 1;" />
  <img src="./docs/arm_mask.png" style="height: 120px; object-fit: contain; flex: 1;" />
  <img src="./docs/filtered_pc.png" style="height: 120px; object-fit: contain; flex: 1;" />
</div>



* Vision-language Feature Extraction

You can directly download our preprocessed files as follows:
| Description | Link | Config |
|----------|----------|----------|
| EgoPAT3D-DT from USST | [EgoPAT3D-postproc](https://pan.sjtu.edu.cn/web/share/f2783a1b3cfca7106175e86f7e089314)  | xx | 
| camera egomotion | [motion_feats](https://pan.sjtu.edu.cn/web/share/383e0314aa1cd3348640d82b10a785f1)  | xx | 
| occupancy voxel grids | [egopat_voxel_filtered](https://pan.sjtu.edu.cn/web/share/db130ef239321f33953074f71157e01e)    | xx |
| MobileSAM requirements | [MobileSAMv2](https://pan.sjtu.edu.cn/web/share/2e043be9b77d84183b2eaa97d26d7efd)     | xx | 
| point cloud example from raw EgoPAT3D | [pointcloud_bathroomCabinet_1](https://pan.sjtu.edu.cn/web/share/f78421cdabf38c5f0fb360232d9249bd)    | xx |


#### 1. Test MMTwin on EgoPAT3D-DT


#### 2. Train MMTwin on EgoPAT3D-DT






### CABH Benchmark

We collected multiple egocentric videos capturing human hands performing simple object manipulation tasks. This benchmark enables rapid validation of the potential of human hand trajectory prediction models for downstream manipulation applications.​​

![](https://github.com/IRMVLab/MMTwin/blob/main/docs/pred_combined.gif)
Past trajectories are shown in green, and MMTwin’s predicted future trajectories are displayed in red. The direct detection results from the visual grounding model on future frames are visualized in blue. As can be seen, MMTwin demonstrates performance comparable to the visual grounding model even if the visual grounding model can "look into future".

Feel free to download the raw/preprocessed data of CABH benchmark.

| Task | Description | Link (raw) | Link (preprocessed) | Link (GLIP feats) | Link (train/test splits) |
|----------|----------|----------|----------|----------|----------|
|    1     |    place the cup on the coaster     |    [hand_data_red_cup.tar.gz](https://pan.sjtu.edu.cn/web/share/921173eaddd9f64c609b78bcd0314174)  |  [hand_data_for_pipeline_mask_redcup.tar.gz](https://pan.sjtu.edu.cn/web/share/56557c9526a9c2faa37150e8eeb1bca3)   | [glip_feats_redcup.tar.gz](https://pan.sjtu.edu.cn/web/share/1cef4958eea97fe41c889111095c18d5)  |  [train_split.txt](https://pan.sjtu.edu.cn/web/share/f9fa399422307f7b3e32bed2f534c8ff)/[test_split.txt](https://pan.sjtu.edu.cn/web/share/548b44a2fbfc0795020d7e51d8b52aa6)    |
|    2     |    put the apple on the plate     |    [hand_data_red_apple.tar.gz](https://pan.sjtu.edu.cn/web/share/ff0e36b5db1e0192d64d5cbfb5597b5c)    |  [hand_data_for_pipeline_mask_redapple.tar.gz](https://pan.sjtu.edu.cn/web/share/064b9fe4e5acaca3408e1293a27eae35)   | [glip_feats_redapple.tar.gz](https://pan.sjtu.edu.cn/web/share/eba393250a4c960a46cb566aaa88c10c)   | [train_split.txt](https://pan.sjtu.edu.cn/web/share/51236f59c34741084eef0e13378ca6ce)/[test_split.txt](https://pan.sjtu.edu.cn/web/share/8eeca488c3212e210b1be65ca25fd128)   |
|    3     |    place the box on the shelf     |    [hand_data_box.tar.gz](https://pan.sjtu.edu.cn/web/share/898718217ac4b8f0640578e38f04b8d2)     |  [hand_data_for_pipeline_mask_box.tar.gz](https://pan.sjtu.edu.cn/web/share/56cacb8a5a65dd71dd6cf304bc6e3f19)   |  [glip_feats_box.tar.gz](https://pan.sjtu.edu.cn/web/share/13b67a41937e61f8048a2a805290834f)  | [train_split.txt](https://pan.sjtu.edu.cn/web/share/338844753b5023ff588f68030226eef9)/[test_split.txt](https://pan.sjtu.edu.cn/web/share/7bcd035fbbb4cff348ab9cbe5d59114f)  |



* **Link (raw)**: Raw RGB and depth images from headset realsense D435i.
* **Link (preprocessed)**: Preprocessed data for HTP. Please refer to `read_cabh.ipynb` for more details.
* **Link (GLIP feats)**: Vision-language features extracted by [GLIP](https://github.com/microsoft/GLIP), a powerful visual grounding model.
* **Link (train/test splits)**: The dataset splits used for training and evaluation.

We also collected an extension of the CABH benchmark to evaluate multi-finger prediction, which will be released soon.

If our work is helpful to your research, we would appreciate a citation to our paper:

```
@misc{ma2025mmtwin,
      title={Novel Diffusion Models for Multimodal 3D Hand Trajectory Prediction}, 
      author={Junyi Ma and Wentao Bao and Jingyi Xu and Guanzhong Sun and Xieyuanli Chen and Hesheng Wang},
      year={2025},
      eprint={2504.07375},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.07375}, 
}
```
