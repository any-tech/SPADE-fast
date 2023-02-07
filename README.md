# Wild SPADE : The Fast and the Furious 🏎🔥

This is an implementation of the paper [Sub-Image Anomaly Detection with Deep
Pyramid Correspondences](https://arxiv.org/pdf/2005.02357.pdf).

We measured accuracy and speed for k=3, k=5 and k=50.

This code was implemented with reference to [SPADE-pytorch](https://github.com/byungjae89/SPADE-pytorch), thanks.

<br/>

## Prerequisites

- faiss-gpu (easy to install with conda : [ref](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md))
- torch
- torchvision
- numpy
- opencv-python
- scipy
- matplotlib
- scikit-learn
- torchinfo
- tqdm


Install prerequisites with:  
```
conda install --file requirements.txt
```

<br/>

If you already download [`MVTec AD`](https://www.mvtec.com/company/research/datasets/mvtec-ad/) dataset, move a file to `data/mvtec_anomaly_detection.tar.xz`.  
If you don't have a dataset file, it will be automatically downloaded during the code running.

When you are ready, it will look like this:
```
./
├── main.py
└── mvtec_anomaly_detection
    ├── bottle
    ├── cable
    ├── capsule
    ├── carpet
    ├── grid
    ├── hazelnut
    ├── leather
    ├── metal_nut
    ├── pill
    ├── screw
    ├── tile
    ├── toothbrush
    ├── transistor
    ├── wood
    └── zipper
```

<br/>

## Usage

To test **SPADE** on `MVTec AD` dataset:
```
python main.py
```

After running the code above, you can see the ROCAUC results in `result/roc_curve.png`

<br/>

## Results

Below is the implementation result of the test set ROCAUC on the `MVTec AD` dataset.  

### 1. Image-level anomaly detection accuracy (ROCAUC %)

| | Paper | This Repo<br/>k=3 | This Repo<br/>k=5 | This Repo<br/>k=50 |
| - | - | - | - | - |
| bottle | - | 96.5 | XX.X | XX.X |
| cable | - | 84.7 | XX.X | XX.X |
| capsule | - | 90.5 | XX.X | XX.X |
| carpet | - | 92.4 | XX.X | XX.X |
| grid | - | 49.5 | XX.X | XX.X |
| hazelnut | - | 89.3 | XX.X | XX.X |
| leather | - | 94.9 | XX.X | XX.X |
| metal_nut | - | 71.2 | XX.X | XX.X |
| pill | - | 79.9 | XX.X | XX.X |
| screw | - | 67.1 | XX.X | XX.X |
| tile | - | 96.7 | XX.X | XX.X |
| toothbrush | - | 86.7 | XX.X | XX.X |
| transistor | - | 90.4 | XX.X | XX.X |
| wood | - | 97.0 | XX.X | XX.X |
| zipper | - | 96.4 | XX.X | XX.X |
| Average | 85.5 | 85.5 | XX.X | XX.X |

<br/>

### 2. Pixel-level anomaly detection accuracy (ROCAUC %)

| | Paper | This Repo<br/>k=3 | This Repo<br/>k=5 | This Repo<br/>k=50 |
| - | - | - | - | - |
| bottle | 98.4 | 97.0 | XX.X | XX.X |
| cable | 97.2 | 92.7 | XX.X | XX.X |
| capsule | 99.0 | 98.2 | XX.X | XX.X |
| carpet | 97.5 | 98.9 | XX.X | XX.X |
| grid | 93.7 | 96.8 | XX.X | XX.X |
| hazelnut | 99.1 | 98.3 | XX.X | XX.X |
| leather | 97.6 | 99.2 | XX.X | XX.X |
| metal_nut | 98.1 | 96.8 | XX.X | XX.X |
| pill | 96.5 | 94.3 | XX.X | XX.X |
| screw | 98.9 | 99.0 | XX.X | XX.X |
| tile | 87.4 | 92.4 | XX.X | XX.X |
| toothbrush | 97.9 | 98.8 | XX.X | XX.X |
| transistor | 94.1 | 87.4 | XX.X | XX.X |
| wood | 88.5 | 94.8 | XX.X | XX.X |
| zipper | 96.5 | 98.3 | XX.X | XX.X |
| Average | 96.0 | 96.2 | XX.X | XX.X |

<br/>

### 3. Processing time (sec)

| | Paper | This Repo<br/>k=3 | This Repo<br/>k=5 | This Repo<br/>k=50 |
| - | - | - | - | - |
| bottle | - | 3.5 | 4.0 | XX.X |
| cable | - | 5.8 | 6.0 | XX.X |
| capsule | - | 4.7 | 5.0 | XX.X |
| carpet | - | 4.3 | XX.X | XX.X |
| grid | - | 2.8 | XX.X | XX.X |
| hazelnut | - | 4.0 | XX.X | XX.X |
| leather | - | 4.4 | XX.X | XX.X |
| metal_nut | - | 4.1 | XX.X | XX.X |
| pill | - | 6.2 | XX.X | XX.X |
| screw | - | 5.5 | XX.X | XX.X |
| tile | - | 4.3 | XX.X | XX.X |
| toothbrush | - | 1.7 | XX.X | XX.X |
| transistor | - | 3.6 | XX.X | XX.X |
| wood | - | 3.1 | XX.X | XX.X |
| zipper | - | 5.5 | XX.X | XX.X |

```
CPU : Intel Xeon Platinum 8360Y
GPU : NVIDIA A100 SXM4
```

<br/>

### ROC Curve 

![roc](./assets/roc_curve.png)

### Localization results  

![bottle](./assets/bottle_000.png)  
![cable](./assets/cable_000.png)  
![capsule](./assets/capsule_000.png)  
![carpet](./assets/carpet_000.png)  
![grid](./assets/grid_000.png)  
![hazelnut](./assets/hazelnut_000.png)  
![leather](./assets/leather_000.png)  
![metal_nut](./assets/metal_nut_000.png)  
![pill](./assets/pill_000.png)  
![screw](./assets/screw_000.png)  
![tile](./assets/tile_000.png)  
![toothbrush](./assets/toothbrush_000.png)  
![transistor](./assets/transistor_000.png)  
![wood](./assets/wood_000.png)  
![zipper](./assets/zipper_000.png)  
