# OnZeta
PyTorch Implementation for Our ECCV'24 Paper: "Online Zero-Shot Classification with CLIP"

## Requirements
* Python 3.9
* PyTorch 1.12
* [CLIP](https://github.com/openai/CLIP)

## Usage:
OnZeta with pre-trained ResNet-50
```
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
python dataset.py
python main_online.py --dataset cifar10 --arch RN50 --repeat 5
```

## Citation
If you use the package in your research, please cite our paper:
```
@inproceedings{qian2024onzeta,
  author    = {Qi Qian and
               Juhua Hu},
  title     = {Online Zero-Shot Classification with CLIP},
  booktitle = {The 18th European Conference on Computer Vision, {ECCV} 2024},
  year      = {2024}
}
