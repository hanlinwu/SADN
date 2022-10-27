## Learning Dynamic Scale Awareness and Global Implicit Functions for Continuous-Scale Super-Resolution of Remote Sensing Images

by Hanlin Wu, Ning Ni, and Libao Zhang, details are in [paper](https://arxiv.org/abs/2210.07598).

## Usage

### Clone the repository:
```
https://github.com/hanlinwu/SADN.git
```
## Requirements:
- pytorch==1.10.0
- pytorch-lightning==1.5.5
- numpy
- opencv-python
- easydict
- tqdm

### Train:

1. Download training datset from this [url](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
2. Change the `train.datapath` and `valid.data_path` in `config/your_config_file.yaml`
3. Do training:
   ```
   python train.py --config config/your_config_file.yaml
   ```

### Test:

1. Download benchmark datasets Set14,B100,Urban100,Manga109, and put them on path: `load/benchmark/datset_name`

2. test without self-ensemble
   ```
   python test.py --checkpoint your_checkpoint_path --datasets Set14,B100,Urban100 --scales 2,3,4
   ```
3. test with self-ensemble
   ```
   python test.py --checkpoint your_checkpoint_path --datasets Set14,B100,Urban100 --scales 2,3,4 --self_ensemble
   ```