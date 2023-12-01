# PowerSkel: A Device-Free Human Skeleton Estimation Framework for Electric Power Operation Based on CSI Signal


## Training
- The example data is in the `./dataset/csi` directory. 
  The data involves power station environment and is collected in cooperation with the power department, the data is confidential. If you need cooperation, please email us.
  If you want to replace your own data, you can make changes according to the form from the given csv file in `./dataset/csi`.
  Email address: cunyiyin1125@gmail.com

- ```shell
  python train.py --root ./dataset/csi/ --model_names conti conti --num_workers 8 --print_freq 10 --gpu-id 0,1,2,3
  ```

## Requirements
Our environment and package versions are:
- python 3.7.16
- pytorch 1.13.1
- 4 NVIDIA RTX A6000 GPUs with 48G memory per GPU


## Acknowledgements
This repo is partly based on the following repos, thank the authors a lot.
- [shaoeric/Online-Knowledge-Distillation-via-Collaborative-Learning](https://github.com/shaoeric/Online-Knowledge-Distillation-via-Collaborative-Learning)
- [gpeyre/SinkhornAutoDiff](https://github.com/gpeyre/SinkhornAutoDiff)

