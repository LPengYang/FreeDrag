<p align="center">
  <img src="./resources/logo2.png">
</p>

# FreeDrag: Point Tracking is Not You Need for Interactive Point-based Image Editing

Official implementation of **FreeDrag: Point Tracking is Not You Need for Interactive Point-based Image Editing**.
- *Authors*: Pengyang Ling*, [Lin Chen*](https://lin-chen.site), Huaian Chen, Yi Jin
- *Institutes*: University of Science and Technology of China; Shanghai AI Laboratory
- [Paper] [Demo] [Project Page]

This repo proposes FreeDrag, a novel interactive point-based image editing framework free of the laborious and unstable point tracking process🔥🔥🔥.

## Abstract
To serve the intricate and varied demands of image editing, precise and flexible manipulation of image content is indispensable. Recently, DragGAN has achieved impressive editing results through point-based manipulation. It includes two alternating steps: (1) a motion supervision step that drives the handle points to move toward the target positions and (2) a point tracking step that consistently localizes the position of moved handle points. However, we have observed that **DragGAN struggles with maintaining content and layout consistency due to laborious and unstable point tracking**. In light of this, we propose a tracking-free point-based image editing framework called **FreeDrag**.
Specifically, for each handle point, FreeDrag decomposes the overall movement toward the final target position into numerous sub-movements toward customized positions calculated based on the "virtual state" of the handle point. This decomposition deliberately controls the difficulty of each point's movement, ensuring steadily reaches the target position. Moreover, an adaptive updating strategy is designed to calculate the "virtual state" of each handle point without point tracking. Extensive experiments demonstrate that FreeDrag enables robust point manipulation in challenging scenarios with similar structures, fine details, or under multi-point targets.

![](resources/fig1.png)

## 📜 News
[2023/7/7] The paper is released!

## 💡 Highlights
- [ ] WebUI of FreeDrag
- [ ] Diffusion-based FreeDrag
- [ ] FreeDrag anything **3D**

## 🛠️Usage
**The demo and detailed code will be released in this or next week, please stay tuned🔥!**

## ❤️Acknowledgments
- [DragGAN](https://github.com/XingangPan/DragGAN/)
- [DragDiffusion](https://yujun-shi.github.io/projects/dragdiffusion.html)
- [StyleGAN3](https://github.com/NVlabs/stylegan3)

## ✒️ Citation
If you find our work helpful for your research, please consider citing the following BibTeX entry.
```bibtex
@article{ling2023free,
  title={FreeDrag: Point Tracking is Not You Need for Interactive Point-based Image Editing},
  author={Pengyang, Ling and Lin, Chen and Huaian, Chen and Yi, Jin},
  journal={arXiv preprint},
  year={2023}
}
```
