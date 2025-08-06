## ![Static Badge](https://img.shields.io/badge/ICCV-2025-blue) Synthesizing Near-Boundary OOD Samples for Out-of-Distribution Detection


Official PyTorch implementation of the iccv 2025 (highlight) paper:

**[Synthesizing Near-Boundary OOD Samples for Out-of-Distribution Detection](https://arxiv.org/abs/2507.10225)**

Jinglun Li, Kaixun Jiang, Zhaoyu Chen, Bo Lin, Yao Tang, Weifeng Ge, Wenqiang Zhang

<p align="center">
  <img src="https://github.com/Jarvisgivemeasuit/SynOOD_infer_pipeline/blob/master/readme_imgs/framework.png" width=100%/>
</p>  

Abstract: *Pre-trained vision-language models have exhibited remarkable abilities in detecting out-of-distribution (OOD) samples. However, some challenging OOD samples, which lie close to in-distribution (InD) data in image feature space, can still lead to misclassification. The emergence of foundation models like diffusion models and multimodal large language models (MLLMs) offers a potential solution to this issue. In this work, we propose SynOOD, a novel approach that harnesses foundation models to generate synthetic, challenging OOD data for fine-tuning CLIP models, thereby enhancing boundary-level discrimination between InD and OOD samples. Our method uses an iterative in-painting process guided by contextual prompts from MLLMs to produce nuanced, boundary-aligned OOD samples. These samples are refined through noise adjustments based on gradients from OOD scores like the energy score, effectively sampling from the InD/OOD boundary. With these carefully synthesized images, we fine-tune the CLIP image encoder and negative label features derived from the text encoder to strengthen connections between near-boundary OOD samples and a set of negative labels. Finally, SynOOD achieves state-of-the-art performance on the large-scale ImageNet benchmark, with minimal increases in parameters and runtime. Our approach significantly surpasses existing methods, and codes are available at https://github.com/Jarvisgivemeasuit/SynOOD.*

## Dataset Preparation

#### In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data (not necessary) and validation data like `./data/ImageNet/train` and  `./data/Imagenet/val`, respectively.

We've released the [inference code](https://github.com/Jarvisgivemeasuit/SynOOD_infer_pipeline) of SynOOD.

## near-OOD Samples Generation

To generate the near-OOD samples of ImageNet, please run:

```
sh generate_near_ood_sample.sh
```

For a complete workflow, refer to `run.sh`, which orchestrates all steps in the SynOOD pipeline.


**We've released the [inference code](https://github.com/Jarvisgivemeasuit/SynOOD_infer_pipeline) of SynOOD.**

**The near-OOD dataset will be released soon.**


## Citation

```
@misc{li2025synthesizingnearboundaryoodsamples,
      title={Synthesizing Near-Boundary OOD Samples for Out-of-Distribution Detection}, 
      author={Jinglun Li and Kaixun Jiang and Zhaoyu Chen and Bo Lin and Yao Tang and Weifeng Ge and Wenqiang Zhang},
      year={2025},
      eprint={2507.10225},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.10225}, 
}
```