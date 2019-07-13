# mdnetmask

## Prerequisites
- python 2.7
- [PyTorch](http://pytorch.org/) and its dependencies 


### Tracking
```bash
 cd tracking
 python run_tracker.py -s DragonBaby [-d (display fig)] [-f (save fig)]

### Pretraining
 - Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat"
 - Download [VOT](http://www.votchallenge.net/) datasets into "dataset/vot201x"
``` bash
 cd pretrain
 python prepro_data.py
 python train_mdnet.py
```
### dataset
- Download [OTB100]http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html

