# MuVi
MuVi: Official repository of "Single Image Test-Time Adaptation via Multi-View Co-Training" In MICCAI 2025.

![architecture](https://github.com/smriti-joshi/muvi/blob/main/images/architecture.png)

## Pipeline Overview

This pipeline is built on top of the nnU-Net inference pipeline, extending the [nnUNetPredictor](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py) class.

⚠️ Note: It currently supports single-fold inference only (not five-fold ensembling).

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
    ├── infer_with_test_time_adaptation.py  # Entry point for test-time adaptation inference
    ├── run.sh                      # Example bash script to run experiments
    ├── utils.py                    # Shared utility functions
    │
    ├── test_time_training_methods/ # Implementations of different TTA methods
    │   ├── bnadapt/                # BatchNorm adaptation method
    │   │   ├── LICENSE
    │   │   ├── NOTICE
    │   │   ├── bn.py               # Core BN adaptation logic
    │   │   └── __pycache__/
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
    │   ├── muvi/                   # Our proposed MuVi method
    │   │   ├── config.yaml         # MuVi hyperparameters & config
    │   │   └── muvi_trainer.py     # MuVi training/adaptation pipeline
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





