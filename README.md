# ESDAN


## A general lightweight image super-resolution with sharpening enhancement and double attention network

## Dependencies

- Python >= 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.5.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb`
- [option] Python packages: [`pip install tensorboardX`](https://github.com/lanpa/tensorboardX), for visualizing curves.


  
## How to Test
1. Clone this github repo. 
```
git clone https://github.com/Czs138/ESDAN
cd ESDAN
```
2. Download the five test datasets (Set5, Set14, B100, Urban100, Manga109) from [Google Drive](https://drive.google.com/drive/folders/1lsoyAjsUEyp7gm1t6vZI9j7jr9YzKzcF?usp=sharing) 

3. Pretrained models have be placed in `./experiments/pretrained_models/` folder. 

4. Run test. 
```
cd codes
python test.py -opt option/test/test_PANx4.yml
```
5. The output results will be sorted in `./results`. 

## How to Train

1. Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://github.com/LimBee/NTIRE2017) from [Google Drive](https://drive.google.com/drive/folders/1B-uaxvV9qeuQ-t7MFiN1oEdA6dKnj2vW?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1CFIML6KfQVYGZSNFrhMXmA)

2. Generate Training patches. Modified the path of your training datasets in `./codes/data_scripts/extract_subimages.py` file.

3. Run Training.

```
python train.py -opt options/train/train_PANx4.yml
```
4. More training commond can be found in `./codes/run_scripts.sh` file.





