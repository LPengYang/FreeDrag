# :boom: Code is available now!

<p align="center">
  <img src="./resources/logo2.png">
</p>

# Video Demo
[![](resources/fig1.png)](https://user-images.githubusercontent.com/58554846/253602157-2f82633b-1c11-4cdb-a21b-deaf1d4209f3.mp4)

# FreeDrag: Point Tracking is Not You Need for Interactive Point-based Image Editing

Official implementation of **FreeDrag: Point Tracking is Not You Need for Interactive Point-based Image Editing**.
- *Authors*: Pengyang Ling*, [Lin Chen*](https://lin-chen.site), [Pan Zhang](https://panzhang0212.github.io/), Huaian Chen, Yi Jin
- *Institutes*: University of Science and Technology of China; Shanghai AI Laboratory
- [[Paper]](https://arxiv.org/abs/2307.04684) [[Project Page]](https://lin-chen.site/projects/freedrag) [Web Demo]

This repo proposes FreeDrag, a novel interactive point-based image editing framework free of the laborious and unstable point tracking process🔥🔥🔥.
https://github.com/LPengYang/FreeDrag/issues/1#issue-1805005537

## Abstract
To serve the intricate and varied demands of image editing, precise and flexible manipulation of image content is indispensable. Recently, DragGAN has achieved impressive editing results through point-based manipulation. 
However, we have observed that DragGAN struggles with miss tracking, where DragGAN encounters difficulty in effectively tracking the desired handle points, and ambiguous tracking, where the tracked points are situated within other regions that bear resemblance to the handle points. To deal with the above issues, we propose **FreeDrag**, which adopts a feature-oriented approach to free the burden on point tracking within the point-oriented methodology of DragGAN. The **FreeDrag** incorporates adaptive template features, line search, and fuzzy localization techniques to perform stable and efficient point-based image editing. Extensive experiments demonstrate that our method is superior to the DragGAN and enables stable point-based editing in challenging scenarios with similar structures, fine details, or under multi-point targets. 

## 📜 News
[2023/7/11] The [paper](https://arxiv.org/abs/2307.04684) and [project page](https://lin-chen.site/projects/freedrag) are released!

## 💡 Highlights
- [ ] WebUI of FreeDrag
- [ ] Diffusion-based FreeDrag
- [ ] FreeDrag anything **3D**

## 🛠️Usage

First download the pre-trained models of stylegan2
```
bash download_models.sh
```
Then initialize the gradio for interactive point-based manipulation

```
export CUDA_LAUNCH_BLOCKING=1
python FreeDrag_gradio.py
```

## ❤️Acknowledgments
- [DragGAN](https://github.com/XingangPan/DragGAN/)
- [DragDiffusion](https://yujun-shi.github.io/projects/dragdiffusion.html)
- [StyleGAN3](https://github.com/NVlabs/stylegan3)

## ✒️ Citation
If you find our work helpful for your research, please consider citing the following BibTeX entry.
```bibtex
@article{ling2023freedrag,
  title={FreeDrag: Point Tracking is Not You Need for Interactive Point-based Image Editing},
  author={Ling, Pengyang and Chen, Lin and Zhang, Pan and Chen, Huaian and Jin, Yi},
  journal={arXiv preprint arXiv:2307.04684},
  year={2023}
}
```

## 🌟 Star History
[![Star History Chart](https://api.star-history.com/svg?repos=LPengYang/FreeDrag&type=Date)](https://star-history.com/#LPengYang/FreeDrag&Date)
