<div align="center">

# Official implementation of the paper "BAC-GCN: Background-Aware CLIP-GCN Framework for Unsupervised Multi-Label Classification" [ACM MM '25]

[![Static Badge](https://img.shields.io/badge/Conference-ACM%20MM%2025-green)](https://doi.org/10.1145/3746027.3755253)
[![ProjectPage](https://img.shields.io/badge/Project_Page-BAC_GCN-blue)](https://yonghyeonjo46.github.io/BAC-GCN/)
[![arXiv](https://img.shields.io/badge/BAC_GCN%20arXiv--blue?logo=arxiv-v1&color=%23B31B1B)](https://arxiv.org/abs/)
<!--[![arXiv](https://img.shields.io/badge/UniDepthV1%20arXiv-2403.18913-blue?logo=arxiv-v1&color=%23B31B1B)](https://arxiv.org/abs/)-->


</div>
This is an official implementation of the ACM Multimedia paper "Official implementation of the paper "BAC-GCN: Background-Aware CLIP-GCN Framework for Unsupervised Multi-Label Classification"

<br> 
<br> 

<div align="center">
  <img src="assets/final_method.png" alt="Final Method">
</div>

## Abstract

Multi-label classification has recently demonstrated promising performance through CLIP-based unsupervised learning. However, existing CLIP-based approaches primarily focus on object-centric features, which limits their ability to capture rich contextual dependencies between objects and their surrounding scenes. In addition, the vision transformer architecture of CLIP exhibits a bias toward the most prominent object, often failing to recognize small or less conspicuous objects precisely. To address these limitations, we propose Background-Aware CLIP-GCN (BAC-GCN), a novel framework that explicitly models class-background interactions and is designed to capture fine-grained visual patterns of small objects effectively. BAC-GCN is composed of three key components: (i) a Similarity Kernel that extracts patch-level local features for each category (i.e., class and background), (ii) a CLIP-GCN that captures relational dependencies between local-global and class-background features, and (iii) a Re-Training for Small Objects (ReSO) strategy that enhances the representation of small and hard-to-learn objects by learning their distinctive visual characteristics. Therefore, our method facilitates a deeper understanding of complex visual contexts, enabling the model to make decisions by leveraging diverse visual cues and their contextual relationships. Extensive experiments demonstrate that BAC-GCN achieves state-of-the-art performance on three benchmark multi-label datasets: VOC07, COCO, and NUS, validating the effectiveness of our approach.

<!--
## YouTube
<div align="center">

<a href="https://www.youtube.com/watch?v=SnWqZ_lb93Y"><img src="https://github.com/user-attachments/assets/86fa69a1-ee69-468f-ac99-d38fcb873934" alt="youtube video" width="600"/></a>

</div>
-->

## How to run

### Create Environment

You can build the environment by following the instruction below.

```bash
# clone project
git clone https://github.com/yonghyeonjo46/BAC-GCN.git
cd BAC-GCN

# create conda environment
cd docker
conda create -n bac python=3.9
conda activate bac

# install pytorch according to instructions
# https://pytorch.org/get-started/
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python ftfy regex tqdm ttach lxml

# install requirements
pip install -r requirements.txt
```

### Prepare Dataset

Please download the datasets from the original sources. Then, please place them as below.

```
data_folder/
├── coco2014/
│   ├── train2014
│   └── val2014
├── voc2007/
│   ├── VOCtrainval_06-Nov-2007
│   └── VOCtest_06-Nov-2007
├── voc2012/
│   ├── VOC2012_test
│   └── VOC2012_train_val
└── nuswide/
    ├── Flickr
    └── ImageList
```

### Prepare Source

Please download the datasets from the original sources. Then, please place them as below.
https://


### Source Training

Training configuration is based on [Hydra](https://hydra.cc). Please see there for the format and instructions on how to use it.

```bash
python src/train.py trainer=gpu experiment=office31_src
```

### Target Training

```bash
python src/train.py trainer=gpu experiment=office31_tgt_ours_pb_teachaug_directed
```

Please see details in: [configs/experiment/](configs/experiment/)

### To see results

The logs are managed by [mlflow](https://mlflow.org).

```bash
cd logs/mlflow

mlflow ui
```
<!--
## Acknowledgement

Our implementation is based on the following works. We greatly appreciate all these excellent works.

+ [AaD](https://github.com/Albert0147/AaD_SFDA)
+ [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template)
+ [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library)
+ [DDA](https://github.com/moskomule/dda)

## Citation

```
@InProceedings{Mitsuzumi_2024_CVPR,
    author    = {Mitsuzumi, Yu and Kimura, Akisato and Kashima, Hisashi},
    title     = {Understanding and Improving Source-free Domain Adaptation from a Theoretical Perspective},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {28515-28524}
}
-->
```
