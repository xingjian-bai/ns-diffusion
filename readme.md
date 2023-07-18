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

### relation
### [inside pipeline] evaluation pipeline
- choose from 1O, 2O, 3O, 4O
- load given model
- combine single image evaluation
- eval on multi-images, use 


