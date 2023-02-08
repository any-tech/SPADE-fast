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
| bottle | - | 96.5 | 96.6 | ９５．７ |
| cable | - | 84.7 | 84.7 | 80.9 |
| capsule | - | 90.5 | 89.7 | 81.8 |
| carpet | - | 92.4 | 92.5 | 91.8 |
| grid | - | 49.5 | 45.7 | 33.0 |
| hazelnut | - | 89.3 | 88.8 | 85.3 |
| leather | - | 94.9 | 94.6 | 92.8 |
| metal_nut | - | 71.2 | 70.0 | 62.1 |
| pill | - | 79.9 | 79.2 | 78.0 |
| screw | - | 67.1 | 65.3 | 49.8 |
| tile | - | 96.7 | 96.4 | 95.8 |
| toothbrush | - | 86.7 | 86.9 | 75.6 |
| transistor | - | 90.4 | 90.0 | 87.4 |
| wood | - | 97.0 | 96.8 | 96.6 |
| zipper | - | 96.4 | 96.4 | 95.6 |
| Average | 85.5 | 85.5 | 84.9 | 80.1 |

<br/>

### 2. Pixel-level anomaly detection accuracy (ROCAUC %)

| | Paper | This Repo<br/>k=3 | This Repo<br/>k=5 | This Repo<br/>k=50 |
| - | - | - | - | - |
| bottle | 98.4 | 97.0 | 97.2 | 97.7 |
| cable | 97.2 | 92.7 | 93.4 | 94.5 |
| capsule | 99.0 | 98.2 | 98.3 | 98.6 |
| carpet | 97.5 | 98.9 | 98.9 | 99.0 |
| grid | 93.7 | 96.8 | 98.2 | 98.6 |
| hazelnut | 99.1 | 98.3 | 98.4 | 98.6 |
| leather | 97.6 | 99.2 | 99.2 | 99.2 |
| metal_nut | 98.1 | 96.8 | 97.0 | 97.3 |
| pill | 96.5 | 94.3 | 94.7 | 95.5 |
| screw | 98.9 | 99.0 | 99.1 | 99.3 |
| tile | 87.4 | 92.4 | 92.7 | 93.7 |
| toothbrush | 97.9 | 98.8 | 98.9 | 98.9 |
| transistor | 94.1 | 87.4 | 88.7 | 90.9 |
| wood | 88.5 | 94.8 | 94.9 | 95.2 |
| zipper | 96.5 | 98.3 | 98.5 | 98.8 |
| Average | 96.0 | 96.2 | 96.5 | 97.1 |

<br/>

### 3. Processing time (sec)

| | Paper | This Repo<br/>k=3 | This Repo<br/>k=5 | This Repo<br/>k=50 |
| - | - | - | - | - |
| bottle | - | 3.5 | 3.6 | 13.2 |
| cable | - | 5.8 | 6.0 | 21.9 |
| capsule | - | 4.7 | 5.0 | 19.7 |
| carpet | - | 4.3 | 4.6 | 17.5 |
| grid | - | 2.8 | 3.1 | 12.3 |
| hazelnut | - | 4.0 | 4.3 | 15.5 |
| leather | - | 4.4 | 5.0 | 18.8 |
| metal_nut | - | 4.1 | 4.5 | 18.5 |
| pill | - | 6.2 | 6.6 | 25.3 |
| screw | - | 5.5 | 6.2 | 23.1 |
| tile | - | 4.3 | 4.9 | 16.3 |
| toothbrush | - | 1.7 | 1.8 | 6.5 |
| transistor | - | 3.6 | 4.1 | 13.2 |
| wood | - | 3.1 | 3.5 | 12.0 |
| zipper | - | 5.5 | 6.1 | 21.8 |
| Average | - | 4.2 | 4.6 | 17.0 |

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
