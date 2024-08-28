## Learning Dynamic Scale Awareness and Global Implicit Functions for Continuous-Scale Super-Resolution of Remote Sensing Images

by Hanlin Wu, Ning Ni, and Libao Zhang, details are in [paper](https://arxiv.org/abs/2110.15655).

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

## Pretrained checkpoints:
1. Model pretrained on the DIV2k dataset: download from this [url](https://github.com/hanlinwu/SADN/releases/download/v1.0.0/sadn_div2k.zip)
   
   **Note: For natural image data test sets such as Set14, B100, etc., to obtain results consistent with those in the paper, please set the `test_Y` attribute in line 82 of test.py to `True`.**
3. Model pretrained on the remote sensing dataset: download from this [url](https://github.com/hanlinwu/SADN/releases/download/v1.0.0/sadn_rs.zip)

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
