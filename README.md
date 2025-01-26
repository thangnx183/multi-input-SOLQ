# Multi-input SOLQ

This repository is based on  of the NeurIPS 2021 paper [SOLQ: Segmenting Objects by Learning Queries](https://arxiv.org/pdf/2106.02351.pdf).

## Introduction 

- This work solved problem of lost accuracy while dealing with image from close distance while inspecting an object. It happen while try to capture more detail part of the object, in our case, user tried to capture a small damage inside a carpart, while we have better view of damage, we lost the sense (front-back, left-right, carpart name,...) to identify which carpart we currently inspecting 
- Propose method : 
  - Use information from normal view (reference image) as addition information 
  - Queries of zoomin image, output of encoder need to cross-attention with information from reference image
  - New set of queries as result of cross-attention then pass to decoder to perform instance segmentation
  - Propose architect to archive segmentation result of both referernce and zoomin image in singel pass

<div style="align: center">
<img src=./figs/multi-input-SOLQ.png/>
</div>




## Citing SOLQ
If you find SOLQ useful in your research, please consider citing:
```bibtex
@article{dong2021solq,
  title={SOLQ: Segmenting Objects by Learning Queries},
  author={Dong, Bin and Zeng, Fangao and Wang, Tiancai and Zhang, Xiangyu and Wei, Yichen},
  journal={NeurIPS},
  year={2021}
}
```
