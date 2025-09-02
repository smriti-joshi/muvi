# MuVi
MuVi: Official repository of "Single Image Test-Time Adaptation via Multi-View Co-Training" In MICCAI 2025.

<p align="center">
  <img src="https://github.com/smriti-joshi/muvi/blob/main/images/architecture.png" alt="architecture" width="750"/>
  <img src="https://github.com/smriti-joshi/muvi/blob/main/images/Result.png" alt="result" width="750"/>
</p>

## Important Links

- [**Method Paper on arxiv**](https://arxiv.org/abs/2506.23705)
- [**Duke-Breast-Cancer-MRI**](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/)  
- [**ISPY1 Dataset**](https://www.cancerimagingarchive.net/collection/ispy1/) 
- [**TCGA-BRCA Dataset**](https://www.cancerimagingarchive.net/collection/tcga-brca/)

## Pipeline Overview

This pipeline is built on top of the nnU-Net inference pipeline, extending the [nnUNetPredictor](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py) class.

```
📂 Repository Structure

├── LICENSE                         # Main license for this repo
├── NOTICE                          # Notices for reused third-party code
├── README.md                       # You’re reading it :)
│
├── images/                         # Figures for README/docs
│   ├── Result.png                  # Example segmentation results
│   └── architecture.png            # Architecture diagram
│
└── scripts/                        # All experiment & training scripts
    │
    ├── compute_metrics.py          # Evaluation metrics (e.g. Dice, Hausdorff, ASSD)
    ├── infer_with_test_time_adaptation.py  # 🌟 Main entry point for running inference w/ MuVi and other methods
    ├── run.sh                      # Example bash script to run experiments
    ├── utils.py                    # Shared utility functions
    │
    ├── test_time_training_methods/ # Implementations of different TTA methods
    │   ├── bnadapt/                # BatchNorm adaptation method
    │   │   ├── LICENSE
    │   │   ├── NOTICE
    │   │   ├── bn.py               # Core BN adaptation logic
    │   │  
    │   ├── intent/                 # InTent method (single-image TTA)
    │   │   ├── LICENSE
    │   │   ├── intent.py           # InTent core logic
    │   │  
    │   ├── memo/                   # MEMO method (TTA with augmentations)
    │   │   ├── LICENSE
    │   │   ├── memo.py             # MEMO implementation
    │   │   └── test_time_augmentation.py  # Augmentation logic for MEMO
    │   │
    │   ├── muvi/                   # 🌟 Our proposed MuVi method
    │   │   ├── config.yaml         # 🌟 MuVi hyperparameters & configs
    │   │   └── muvi_trainer.py     # 🌟 MuVi training & test-time adaptation logic

    │   │
    │   └── tent/                   # Tent method (entropy minimization)
    │       ├── LICENSE
    │       ├── README.md           # Original Tent docs
    │       ├── cfgs/               # Tent configs
    │       │   ├── norm.yaml
    │       │   ├── source.yaml
    │       │   └── tent.yaml
    │       ├── cifar10c.py         # Dataset helper for corruption benchmarks
    │       ├── conf.py             # Config management
    │       ├── norm.py             # Norm layers for Tent
    │       ├── tent.py             # Tent core implementation
    │       └── requirements.txt    # Tent-specific requirements
```

## How to run?

### 1. Install nnU-Net
Follow the official nnU-Net installation instructions:  
👉 [nnU-Net GitHub](https://github.com/MIC-DKFZ/nnUNet)

> ⚠️ Our pipeline is built on **nnU-Net’s inference pipeline** (`nnUNetPredictor` class).  
> Currently, only **one-fold inference** is supported (not five-fold).

### 2. Modify nnU-Net normalization
To switch between **InstanceNorm** and **BatchNorm**, edit ```norm_op``` in ```plans.json``` file in the respective [nnUNet_results](https://github.com/smriti-joshi/muvi/tree/main/scripts/nnUNet_results/Duke101Baseline) folder. 

### 3. Clone the repository
```
git clone https://github.com/your-username/muvi-tta.git
cd muvi-tta/scripts
```

### 4. Run inference with MuVi

```
python infer_with_test_time_adaptation.py \
    --input /path/to/images \
    --output /path/to/save/results \
    --model /path/to/nnunet_trained_model \
    --method muvi
```

### 5. Compute Evaluation Metrics

```
python compute_metrics.py \
    --predictions /path/to/save/results \
    --ground_truth /path/to/gt_labels
```

```run.sh``` compiles the commands above! 

## Citation

If you use MuVi in your work, we’d love it if you gave us a shout-out by citing our paper!

```
@misc{joshi2025singleimagetesttimeadaptation,
      title={Single Image Test-Time Adaptation via Multi-View Co-Training}, 
      author={Smriti Joshi and Richard Osuala and Lidia Garrucho and Kaisar Kushibar and Dimitri Kessler and Oliver Diaz and Karim Lekadir},
      year={2025},
      eprint={2506.23705},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.23705}, 
}
```
## Acknowledgements

This work builds upon nnUNet repository and adapts several test time adaptation methods to this framework:

- [nnUNet (official)](https://github.com/MIC-DKFZ/nnUNet) 
- [Tent (official)](https://github.com/DequanWang/tent) 
- [InTent (official)](https://github.com/mazurowski-lab/single-image-test-time-adaptation)  
- [MEMO (official)](https://github.com/zhangmarvin/memo)
- [BNAdapt (official)](https://github.com/bethgelab/robustness)
  
We acknowledge the authors of these works for releasing their code.

This repository was developed at the [BCN-AIM](https://www.bcn-aim.org/), [Universitat de Barcelona].  

We gratefully acknowledge support from [RadioVal](https://radioval.eu/) (European research and innovation programme grant agreement No 101057699).

<p align="left">
  <img src="https://github.com/smriti-joshi/muvi/blob/main/images/bcn-aim-logo.png" alt="bcn-aim" width="250"/>
  <img src="https://github.com/smriti-joshi/muvi/blob/main/images/radioval-logo.png" alt="radioval" width="250"/>
  <img src="https://github.com/smriti-joshi/muvi/blob/main/images/Logo_Universitat_de_Barcelona.png" alt="ub" width="250"/>
</p>





