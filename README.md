# Semi-Supervised GAN (Improved GAN) in PyTorch

See `requirements.txt` for package versions. A Jupyter notebook `ssgan_notebook.ipynb` is also provided with Bokeh plots of live updated training metrics.

> Note: Implementation and hyperparameters used may vary slightly from what the papers describe. 

## Example usage

```
python ssgan_exp.py --dataset mnist --epochs 10 --perc_labeled 0.00167 --lr 0.003 --noise_dist uniform --use_weight_norm

python ssgan_exp.py --dataset mreo --epochs 100 --perc_labeled 0.08 --lr 0.0006 --noise_dist normal --no_eq_union
```

## Results and models

| Dataset | Labeled training data | Accuracy | Reference    | Checkpoint |
| :-----: | :------------: | :------: | :----------: | :--------: |
| MNIST   | 100 samples    | 0.9509   | [1], see Table 1 | [model](checkpoints/e10_ckpt_ssgan_mnist_perclabeled0,00167_noisesize100_noiseuniform_lr0,003_featmatch1_weightnorm1_gfaNone_equnion1_seed1000.pth) |
| MREO    | 8%             | 0.8658   | [2], see Table 1 | [model](checkpoints/e100_ckpt_ssgan_mreo_perclabeled0,08_noisesize100_noisenormal_lr0,0006_featmatch1_weightnorm0_gfaNone_equnion0_seed1000.pth) |

## Getting data
MNIST data is used from `torchvision.datasets`.

MREO data can be downloaded from [here](https://github.com/Healthcare-Robotics/mr-gan#download-the-mreo-dataset), we use the compact version.

```
tar -xvf data_processed_compact.tar.gz mreo_data/
```

## References
[1]: Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen, ["Improved Techniques for Training GANs"](https://arxiv.org/abs/1606.03498), 2016.

[2]: Z. Erickson, S. Chernova, and C. C. Kemp, ["Semi-Supervised Haptic Material Recognition for Robots using Generative Adversarial Networks"](https://arxiv.org/abs/1707.02796), 2017.

## Changelog
- (Dec. 28, 2019) Update to PyTorch 1.3. Add results for MNIST and MREO. Add weight normalization, easier setting of hyperparameters, and data loading improvements. 
- (Dec. 26, 2019) Fix bug in labeled loss function, now properly indexes prediction probabilities
