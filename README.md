# Rethinking Class Orders and Transferability in Class Incremental Learning

This is the official repository for the _paper Rethinking Class Orders and Transferability in Class Incremental Learning_. [[Pre-print]](https://www.sciencedirect.com/science/article/abs/pii/S0167865522002252)[[Supp]](https://ars.els-cdn.com/content/image/1-s2.0-S0167865522002252-mmc1.pdf)

## 1. Requirements

The code is implemented in Python 3.6.

As for CUDA, we use CUDA 10.1 and cuDNN 7.6.

For requirements for the Python modules, you can simply run (note that pip version should be >19.0 (or >20.3 for macOS)):

``pip install -r requirements.txt``

## 2. Datasets

For **Group ImageNet**, download the ImageNet64x64 ([[Google Drive]](https://drive.google.com/drive/folders/1ESWOB1C7pHjNOH12Uo0LQXPGnv7MbvPo?usp=sharing) [[百度网盘 (提取码:nqi7)]](https://pan.baidu.com/s/1KlXN-id7ybXn-zYZJv06BA) dataset first and change `data_path` in _imagenet64x64.py_ to your folder path. ImageNet64x64 is a downsampled ImageNet according to https://patrykchrabaszcz.github.io/Imagenet32/.

For **iNaturalist**, download the iNaturalist64x64 ([[Google Drive]](https://drive.google.com/file/d/1jtYs2gB0hv_eXiPzvTkbv-lW2c970tqI/view?usp=sharing) [[百度网盘 (提取码:aksa)]](https://pan.baidu.com/s/1gStNX2gUML1pO2OhGytnXQ) dataset first and change `TRAIN_DIR`/`TEST_DIR` in _inaturalist64x64.py_ to your folder path. iNaturalist64x64 is a cropped and resized version according to the bounding box annotations provided by iNaturalist.

## 3. Usage

### 3.1 Comparisons between the Even and Group class orders

After downloading the datasets and changing the dataset paths, simply run the following scripts for **Group ImageNet** and **Group iNaturalist** respectively (some configurations in the script should be set in advance, e.g. ``LD_LIBRARY_PATH``):

`bash scripts/imagenet64x64_order_even.sh`

`bash scripts/imagenet64x64_order_group.sh`

`bash scripts/inat_order_even.sh`

`bash scripts/inat_order_group.sh`

By default it will run 5 different class orders, and it may take too much time to finish training. For acceleration, you can simply change the ``for i in {1..5}`` to ``for i in {1..1}`` to reduce the number of class orders.

After running the previous scripts, you can reproduce the first two columns in Table 1 by running the following script:

`bash scripts/imagenet64x64_display_accs.sh`

And Table 1 in the supplementary material by:

`bash scripts/inat_display_accs.sh`

### 3.2 Transferability of Even and Group



### 3.3 Greedy order obtained by COSA



## Citation

If you use these codes or find anything that inspires your works, please cite our paper:

```bibtex
@article{he2022rethinking,
  title={Rethinking class orders and transferability in class incremental learning},
  author={He, Chen and Wang, Ruiping and Chen, Xilin},
  journal={Pattern Recognition Letters},
  volume={161},
  pages={67--73},
  year={2022},
  publisher={Elsevier}
}
```

## Contact

If you have any questions when running our code, feel free to contact chen.he@vipl.ict.ac.cn