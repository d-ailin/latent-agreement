# Model Agreement
Repo for the paper: [Great Models Think Alike: Improving Model Reliability via Inter-Model Latent Agreement (ICML 2023)](https://arxiv.org/pdf/2305.01481.pdf)

# Environment Setup

## Install with Command
```
    conda create -n [your_env] python=3.7
    conda activate [your_env]

    bash install.sh
    # or if you want to install only cpu version
    bash install_cpu.sh
```

## Quick Test
Unzip our provided features from [link](https://drive.google.com/file/d/1Xn7fNZEKSy413irBhEJVYi8Hcg80zsV5/view?usp=sharing) in `./saved/extracted_features/cub200` and run the following command:
```
    bash scripts/exc.sh
```
If the environment is all set, you would see the `res/` directory is created and the results are saved in it after running the command above.


# Customize Your Test

## Env Config
```
    # env_config.yaml
    feature_dir: [your_feature_dir]
```
You could follow the format of saved features provided in `./saved/extracted_features/cub200`

**Notice:** samples in `train.pt` should have same order with the index order in `train_idx.npy` in our provided code. You could also change the 'read-in' part in code for your need.


## Exp Config
The running configs are in `conf/`.
```
  | - run
  |  | - pretrain  # set your pretrain large model configs
  |  | - base_model # set your supervised model configs, feel free to change the `name` to your saved model feature file name
  |  | - dataset # set your dataset configs
```

## Run Config
You could check the demo in `scripts/exc.sh` or more specific entries in `scripts/sets/cub200.sh`.



## Citation
If you find this paper or repo useful for your research, please consider citing the paper
```

@InProceedings{deng23great,
  title = 	 {Great Models Think Alike: Improving Model Reliability via Inter-Model Latent Agreement},
  author =       {Deng, Ailin and Xiong, Miao and Hooi, Bryan},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {7675--7693},
  year = 	 {2023},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/deng23f/deng23f.pdf},
  url = 	 {https://proceedings.mlr.press/v202/deng23f.html},
}

```

