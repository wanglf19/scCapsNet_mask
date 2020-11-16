## scCapsNet-mask: an automatic version of scCapsNet

This repository contains the official Keras implementation of:

**scCapsNet-mask: an automatic version of scCapsNet**


**Requirements**
- Python 3.6
- conda 4.4.10
- keras 2.2.4
- tensorflow 1.11.0

**Citation for those code**
- Wang, L., Nie, R., Yu, Z. et al. An interpretable deep-learning architecture of capsule networks for identifying cell-type gene expression programs from single-cell RNA-sequencing data. Nat Mach Intell 2, 693–703 (2020). https://doi.org/10.1038/s42256-020-00244-4
- Wang, L., Zhang, J., and Cai, J. scCapsNet-mask: an automatic version of scCapsNet. Biorxiv 2020.11.02.365346v1. https://doi.org/10.1101/2020.11.02.365346

**1. Model training and analysis**

- *About this article*
```
#Augments:
#'--inputdata', type=str, default='data/PBMC_data.npy', help='address for input data'
#'--inputcelltype', type=str, default='data/PBMC_celltype.npy', help='address for celltype label'
#'--num_classes', type=int, default=8, help='number of cell type'
#'--randoms', type=int, default=30, help='random number to split dataset'
#'--dim_capsule', type=int, default=16, help='dimension of the capsule'
#'--batch_size', type=int, default=400, help='training parameters_batch_size'
#'--epochs', type=int, default=15, help='training parameters_epochs'
#'--training', type=str, default='T', help='training model(T) or loading model(F) '
#'--weights', type=str, default='data/PBMC_demo.weight', help='trained weights'
#'--plot_direction', type=str, default='one_side', help='display option, both_side or one_side'
#'--pc_slice', type=int, default=20, help='fineness divided along PC direction '
#'--threshold', type=float, default=0.05, help='threshold for setting dotted line'


To unzip the PBMC_data.rar and retina_data.rar into dictionary 'data' then run:

For PBMC_dataset
python scCapsNet_mask.py --inputdata=data/PBMC_data.npy --inputcelltype=data/PBMC_celltype.npy --num_classes=8 --dim_capsule=16 --pc_slice=30 --epochs=10

For RBC_dataset
python scCapsNet_mask.py --inputdata=data/retina_data.npy --inputcelltype=data/retina_celltype.npy --num_classes=15 --dim_capsule=32 --pc_slice=20

```

- *Further Explore*
```
#- Testing your own dataset
Tips: set the dim_capsule twice as the number of cell types
python scCapsNet_mask.py --inputdata=your_data --inputcelltype=your_inputcelltype --num_classes=your_num_classes --dim_capsule=your_dim_capsule
```

**2. Model analysis**

- *Demo -- About this article*

The following codes could reproduce Figures in the main text.
```
For PBMC_dataset
python scCapsNet_mask.py --inputdata=data/PBMC_data.npy --inputcelltype=data/PBMC_celltype.npy --num_classes=8 --dim_capsule=16 --pc_slice=30 --weights=data/PBMC_demo.weight --training=F

For RBC_dataset
python scCapsNet_mask.py --inputdata=data/retina_data.npy --inputcelltype=data/retina_celltype.npy --num_classes=15 --dim_capsule=32 --pc_slice=20 --weights=data/retina_demo.weight --training=F
```

**Output**
The output is in the results folder, including
- *two plot*
```
Prediction_accuracy_curve.png (Line_plot)
Choosen_genes.png (Scatter plot)
```

- *training weight*
```
training_n8_r30_dim16_e10_b400_.weight
```

- *Cell type related genes*
```
total_select_genes_one_side.npy
```

**capsule networks implementation**

the capsule parts refer to https://github.com/bojone/Capsule and https://kexue.fm/archives/5112