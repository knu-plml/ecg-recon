# Twelve-Lead ECG Reconstruction from Single-Lead Signals Using Generative Adversarial Networks
### [Paper](!!링크필요)

 Jinho Joo\*<sup>1</sup>,
 Gihun Joo\*<sup>1</sup>,
 Yeji Kim<sup>2</sup>,
 Moo-Nyun Jin<sup>2</sup>,
 Junbeom Park\*\*<sup>2</sup>,
 Hyeonseung Im\*\*<sup>1</sup>  
 <sup>1</sup>Kangwon National University, <sup>2</sup>Ewha Womans University Medical Center  
  \*denotes equal contribution,
  \**denotes corresponding authors  
in MICCAI 2023 ('ORAL or Poster')

<p align="center"><img src='imgs/fig-rbbbaf2-color.png' width="85%" height="85%"></p>
<p align="center"><img src='imgs/fig-rbbbaf2-color-detail.png' width="85%" height="85%"></p>

## Our goal
We propose a novel generative adversarial network that can faithfully reconstruct 12-lead ECG signals from single-lead signal.
Our method can reconstruct 12-lead ECG with CVD-related characteristics effectively.
Thus, our method can be used to bridge commonly available wearable devices that can measure only Lead I and high-performance deep learning-based prediction models using 12-lead ECGs.

## EKGAN Architecture
<p align="center"><img src='imgs/fig-ekgan-new.png' width="85%" height="85%"></p>

## Others
We implemented not only EKGAN but also [Pix2pix](https://arxiv.org/pdf/1611.07004.pdf), [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf), and [CardioGAN](https://arxiv.org/pdf/2010.00104.pdf) with minor modifications so that they can be applied to ECG data. 


