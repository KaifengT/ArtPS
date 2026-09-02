# Category-Level Articulated Object Pose Estimation via Pose-Shape Hypothesis Generation and Verification

Kaifeng Tang · Chi Xu* · Xin Ao · Yuting Ge · Tingrui Guo · Jun Zhou

China University of Geosciences, Wuhan

**ECCV 2026**

[![Project Page](https://img.shields.io/badge/Project-Page-4c8bf5?style=flat-square)](https://github.com/KaifengT/ArtPS)
![Code](https://img.shields.io/badge/Code-Coming%20Soon-orange?style=flat-square)

## Overview

![Overview of the pose-shape hypothesis generation and verification framework](./assets/main_figure.png)

Given a single-view point-cloud observation, our framework generates a set of coupled pose-shape hypotheses, recovers clean part-level geometry when needed, and verifies each hypothesis using joint and shape consistency. The final output retains geometrically plausible pose-shape pairs instead of collapsing occlusion-induced ambiguity into a single prediction.

## Demo

![Qualitative articulated object pose estimation results](./assets/Vis_Pose.webp)



## Highlights

- **Joint pose-shape generation:** models entangled pose and shape uncertainty and samples paired hypotheses from a shared condition.
- **Part-level denoising:** completes and cleans partial real-world point clouds before verification.
- **Explicit geometric verification:** combines joint consistency with an SDF-based cross-space shape score to prune implausible hypotheses.
- **Single-view inference:** estimates per-part 6D poses and canonical object shape from one partial point-cloud observation.


## Code

**Code status:** The source code and pretrained models are **coming soon**.

Training, evaluation, and inference instructions will be added when the code is released.

## Citation

The BibTeX entry will be added after the publication record is available.

## Acknowledgements

We thank the authors of **HOI4D**, **ANCSH**, **OMAD**, **RPF**, and **PPF-Tracker** for their open-source contributions.

This work was supported by the National Natural Science Foundation of China under Grant No. 62273318 and, in part, by the Major Program of Xiangjiang Laboratory under Grant No. 25XJJB01.
