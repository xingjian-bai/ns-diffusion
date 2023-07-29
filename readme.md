# Xingjian
## Baselines
### cfg_bbox3
Adapted from Jiayuan's code. Not factorized, has bugs in size normalization
### cfg_bbox4 - Single Bbox generation on CLEVR
- Refactorized single bbox generation.
- Expect to have no bugs. All generated boxes are square and the sizes match.
### cfg_bbox5 - Two Bbox generation on CLEVR
- models [Denoise, Bidenoise] -> Diffuser[GaussianDiffusion1D] -> trainer[Trainer1D] -> train.py

### cfg_image1 - Single Object generation on CLEVR
- not finished

### cfg_bbox6 - Two Bbox generation on CLEVR, with multi-graphs
- change the dataset: input all relations, instead of one-per-pair
- Will make cfg_bbox*, *<6, deprecated.
- allow batch_size > 1, but use for-loops.

#### [inside pipeline] evaluation pipeline
- choose from 1O, 2O, 3O, 4O
- load given model
- combine single image evaluation
- eval on multi-images, use 


### bbox_classifier
- classify relations based on bboxes.
- Trained on multi-object scenes, as the core part of bbox-evaluation pipeline


## New start: bbox-to-image generation
### cond_image0
- unconditional generation
### cond_image1
- conditioned on object feature and bounding box.
- Train 1-obj scene; train on 2-obj-scene with composition.

### cond_image2
- conditioned on object, or relation, or baoth
- added a global diffusion to train together