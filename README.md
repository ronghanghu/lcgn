# Language-Conditioned Graph Networks for Relational Reasoning

This repository contains the code for the following paper:

* R. Hu, A. Rohrbach, T. Darrell, K. Saenko, *Language-Conditioned Graph Networks for Relational Reasoning*. in ICCV 2019 ([PDF](https://arxiv.org/pdf/1905.04405.pdf))
```
@inproceedings{hu2019language,
  title={Language-Conditioned Graph Networks for Relational Reasoning},
  author={Hu, Ronghang and Rohrbach, Anna and Darrell, Trevor and Saenko, Kate},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

Project Page: http://ronghanghu.com/lcgn

**This is the (original) TensorFlow implementation of LCGN. A PyTorch implementation is available in the [PyTorch branch](https://github.com/ronghanghu/lcgn/tree/pytorch).**

## Installation

1. Install Python 3 (Anaconda recommended: https://www.continuum.io/downloads).
2. Install TensorFlow (we used TensorFlow 1.12.0 in our experiments):  
`pip install tensorflow-gpu`  (or `pip install tensorflow-gpu==1.12.0` to install TensorFlow 1.12.0)
3. Install PyTorch (needed only in CLEVR and CLEVR-Ref+ experiments, and only for feature extraction):  
`pip install torch torchvision`
4. Install a few other dependency packages (NumPy, HDF5, YAML):   
`pip install numpy h5py pyyaml`
5. Download this repository or clone with Git, and then enter the root directory of the repository:  
`git clone https://github.com/ronghanghu/lcgn.git && cd lcgn`

## Train and evaluate on the CLEVR dataset and the CLEVR-Ref+ dataset

The LCGN model is applied to the `14 x 14 x 1024` ResNet-101-C4 features from CLEVR and CLEVR-Ref+.

On CLEVR, we train on the training split and evaluate on the validation and test splits. It gets the following performance on the validation (`val`) and the test (`test`) split of the CLEVR:

| Accuracy on `val` | Accuracy on `test` | Pre-trained model |
| ------------- | ------------- | ------------- |
| 97.90% | 97.88% | [download](https://people.eecs.berkeley.edu/~ronghang/projects/lcgn/pretrained_models_clevr_vqa_ref/lcgn/) |

On CLEVR-Ref+, following the [IEP-Ref](https://github.com/ccvl/iep-ref) code, we cross-valid on the training set and evaluate on the validation set. It gets the following performance on the validation (`locplus_val` in our code) split of the CLEVR-Ref+:

| Accuracy on `locplus_val` | Pre-trained model |
| ------------- | ------------- |
| 74.82% | [download](https://people.eecs.berkeley.edu/~ronghang/projects/lcgn/pretrained_models_clevr_vqa_ref/lcgn_ref/) |

### Download and preprocess the data

1. Download the CLEVR dataset from http://cs.stanford.edu/people/jcjohns/clevr/, and symbol link it to `exp_clevr/clevr_dataset`. After this step, the file structure should look like
```
exp_clevr/clevr_dataset/
  images/
    train/
      CLEVR_train_000000.png
      ...
    val/
    test/
  questions/
    CLEVR_train_questions.json
    CLEVR_val_questions.json
    CLEVR_test_questions.json
  ...
```

If you want to run any experiments on the CLEVR-Ref+ dataset for the referential expression comprehension task, you can download it from [here](https://cs.jhu.edu/~cxliu/2019/clevr-ref+), and symbol link it to `exp_clevr/clevr_locplus_dataset`. After this step, the file structure should look like
```
exp_clevr/clevr_locplus_dataset/
  images/
    train/
      CLEVR_train_000000.png
      ...
    val/
    test/
  refexps/
    clevr_ref+_train_refexps.json
    clevr_ref+_val_refexps.json
  scenes/
    clevr_ref+_train_scenes.json
    clevr_ref+_val_scenes.json
```

2. Build imdbs for the datasets
```
cd exp_clevr/data/
# build imdbs
python build_clevr_imdb.py
python build_clevr_locplus_imdb.py   # only needed if you want to run on CLEVR-Ref+
cd ../../
```

3. Extract ResNet-101-C4 features for the CLEVR images

Here, we use Justin Johnson's [CLEVR feature extraction script](https://github.com/facebookresearch/clevr-iep/blob/master/scripts/extract_features.py), which requires PyTorch.
```
mkdir -p exp_clevr/data/features/spatial/
python exp_clevr/data/extract_features.py \
  --input_image_dir exp_clevr/clevr_dataset/images/train \
  --output_h5_file exp_clevr/data/features/spatial/train.h5
python exp_clevr/data/extract_features.py \
  --input_image_dir exp_clevr/clevr_dataset/images/val \
  --output_h5_file exp_clevr/data/features/spatial/val.h5
python exp_clevr/data/extract_features.py \
  --input_image_dir exp_clevr/clevr_dataset/images/test \
  --output_h5_file exp_clevr/data/features/spatial/test.h5
```

### Training on CLEVR and CLEVR-Ref+

Note:
* By default, the scripts below use a single GPU (GPU 0). To run on multiple or different GPUs, append `GPUS` parameter to the commands below (e.g. appending `GPUS 1` to use GPU 1, or `GPUS 0,1` to use both GPU 0 & 1).
* During training, the script will save the snapshots under `exp_clevr/tfmodel/{exp_name}/`, where `{exp_name}` is one of `lcgn` (for CLEVR) or `lcgn_ref` (for CLEVR-Ref+).

Pretrained models:
* You may skip the training step and directly download the pre-trained models from the links at the top. The downloaded models should be saved under `exp_clevr/tfmodel/{exp_name}/`, where `{exp_name}` is one of `lcgn` (for CLEVR) or `lcgn_ref` (for CLEVR-Ref+).

Training steps：  

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  
1. Train on CLEVR (for VQA):  
`python exp_clevr/main.py --cfg exp_clevr/cfgs/lcgn.yaml train True`
2. Train on CLEVR-Ref+ (for REF):  
`python exp_clevr/main_ref.py --cfg exp_clevr/cfgs_ref/lcgn_ref.yaml train True`

### Testing on CLEVR and CLEVR-Ref+

Note:
* As there is no ground-truth answers for *test* split in the downloaded CLEVR data, when testing on the CLEVR test set **the displayed accuracy will be zero**. You may email the prediction outputs in `exp_clevr/results/lcgn/` to the CLEVR dataset authors for the *test* split accuracy.
* By default, the scripts below use a single GPU (GPU 0). To run on multiple or different GPUs, append `GPUS` parameter to the commands below (e.g. appending `GPUS 1` to use GPU 1, or `GPUS 0,1` to use both GPU 0 & 1).

Testing steps:   

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  
1. Test on CLEVR (for VQA):  
    - test locally on the `val` split:   
    `python exp_clevr/main.py --cfg exp_clevr/cfgs/lcgn.yaml`  
    - test locally on the `test` split (the displayed accuracy will be zero; use prediction outputs in `exp_clevr/results/lcgn/`):   
    `python exp_clevr/main.py --cfg exp_clevr/cfgs/lcgn.yaml TEST.SPLIT_VQA test TEST.DUMP_PRED True`  
2. Test on CLEVR-Ref+ (for REF):  
    - test locally on the `val` split:   
    `python exp_clevr/main_ref.py --cfg exp_clevr/cfgs_ref/lcgn_ref.yaml`  

## Train and evaluate on the GQA dataset

The LCGN model is applicable to three types of features from GQA:
* spatial features: `7 x 7 x 2048` ResNet spatial features (from the GQA dataset release)
* objects features: `N_{det} x 2048` Faster R-CNN ResNet object features (from the GQA dataset release)
* "perfect-sight" object names and attributes: obtained from the **ground-truth** scene graphs in GQA at **both training and test time**. This setting uses two one-hot vectors to represent each object's class name and attributes, and concatenate them as its visual feature. *It does not use the relation annotations in the scene graphs.*

It gets the following performance on the validation (`val_balanced`), the test-dev (`testdev_balanced`) and the test (`test_balanced`) split of the GQA dataset:

| Visual Feature Type  | Accuracy on `val_balanced` | Accuracy on `testdev_balanced` | Accuracy on `test_balanced` (obtained on EvalAI Phase: `test2019`) | Pre-trained model |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| spatial features | 55.29% | 49.47% | 49.21% | [download](https://people.eecs.berkeley.edu/~ronghang/projects/lcgn/pretrained_models_gqa/lcgn_spatial/) |
| objects features | 63.87% | 55.84% | 56.09% | [download](https://people.eecs.berkeley.edu/~ronghang/projects/lcgn/pretrained_models_gqa/lcgn_objects/) |
| "perfect-sight" object names and attributes | 90.23% | n/a* | n/a* | [download](https://people.eecs.berkeley.edu/~ronghang/projects/lcgn/pretrained_models_gqa/lcgn_scene_graph/) |

*This setting requires using the GQA ground-truth scene graphs at both training and test time (only the object names and attributes are used; their relations are not used). Hence, it is not applicable to the test or the challenge setting.

Note: we also release our simple but well-performing "single-hop" baseline for the GQA dataset in a [standalone repo](https://github.com/ronghanghu/gqa_single_hop_baseline). This "single-hop" model can serve as a basis for developing more complicated models.

### Download the GQA dataset

Download the GQA dataset from https://cs.stanford.edu/people/dorarad/gqa/, and symbol link it to `exp_gqa/gqa_dataset`. After this step, the file structure should look like
```
exp_gqa/gqa_dataset
    questions/
        train_all_questions/
            train_all_questions_0.json
            ...
            train_all_questions_9.json
        train_balanced_questions.json
        val_all_questions.json
        val_balanced_questions.json
        submission_all_questions.json
        test_all_questions.json
        test_balanced_questions.json
    spatial/
        gqa_spatial_info.json
        gqa_spatial_0.h5
        ...
        gqa_spatial_15.h5
    objects/
        gqa_objects_info.json
        gqa_objects_0.h5
        ...
        gqa_objects_15.h5
    sceneGraphs/
        train_sceneGraphs.json
        val_sceneGraphs.json
    images/
        ...
```

Note that on GQA images are not needed for training or evaluation -- only questions, features and scene graphs (if you would like to run on the "perfect-sight" object names and attributes) are needed.

### Training on GQA

Note:
* Our models are trained on the GQA `train_balanced` split, which takes a few hours on our machines.  
* By default, the scripts below use a single GPU (GPU 0). To run on multiple or different GPUs, append `GPUS` parameter to the commands below (e.g. appending `GPUS 1` to use GPU 1, or `GPUS 0,1` to use both GPU 0 & 1).
* During training, the script will save the snapshots under `exp_gqa/tfmodel/{exp_name}/`, where `{exp_name}` is one of `lcgn_spatial`, `lcgn_objects` or `lcgn_scene_graph`.

Pretrained models:
* You may skip the training step and directly download the pre-trained models from the links at the top. The downloaded models should be saved under `exp_gqa/tfmodel/{exp_name}/`, where `{exp_name}` is one of `lcgn_spatial`, `lcgn_objects` or `lcgn_scene_graph`.

Training steps：  

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  
1. Train with spatial features (convolutional grid features):  
`python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_spatial.yaml train True`
2. Train with objects features (from detection):  
`python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_objects.yaml train True`
3. Train with "perfect-sight" object names and attributes (one-hot embeddings):  
`python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_scene_graph.yaml train True`

### Testing on GQA

Note:
* The evaluation script below will print out the final VQA accuracy only on when testing on `val_balanced` or `testdev_balanced` split.   
* When running test on the `submission_all` split to generate the prediction file, **the displayed accuracy will be zero**, but the prediction file will be correctly generated under `exp_gqa/results/{exp_name}/` (where `{exp_name}` is one of `lcgn_spatial`, `lcgn_objects` or `lcgn_scene_graph`), and the prediction file path will be displayed at the end. *It takes a long time to generate prediction files.*  
* By default, the scripts below use a single GPU (GPU 0). To run on multiple or different GPUs, append `GPUS` parameter to the commands below (e.g. appending `GPUS 1` to use GPU 1, or `GPUS 0,1` to use both GPU 0 & 1).

Testing steps:   

0. Add the root of this repository to PYTHONPATH: `export PYTHONPATH=.:$PYTHONPATH`  
1. Test with spatial features:  
    - test locally on the `val_balanced` split:   
    `python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_spatial.yaml TEST.SPLIT_VQA val_balanced`  
    - test locally on the `testdev_balanced` split:   
    `python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_spatial.yaml TEST.SPLIT_VQA testdev_balanced`  
    - generate the submission file on `submission_all` for EvalAI (the displayed accuracy will be zero; this takes a long time):   
    `python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_spatial.yaml TEST.SPLIT_VQA submission_all TEST.DUMP_PRED True`
2. Test with objects features:  
    - test locally on the `val_balanced` split:   
    `python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_objects.yaml TEST.SPLIT_VQA val_balanced`  
    - test locally on the `testdev_balanced` split:   
    `python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_objects.yaml TEST.SPLIT_VQA testdev_balanced`  
    - generate the submission file on `submission_all` for EvalAI (the displayed accuracy will be zero; this takes a long time):   
    `python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_objects.yaml TEST.SPLIT_VQA submission_all TEST.DUMP_PRED True`
3. Test with "perfect-sight" object names and attributes (one-hot embeddings):  
    - test locally on the `val_balanced` split:   
    `python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_scene_graph.yaml TEST.SPLIT_VQA val_balanced`  
    - test locally on the `testdev_balanced` split (**This won't work unless you have a file `testdev_sceneGraphs.json` under `exp_gqa/gqa_dataset/sceneGraphs/` that contains scene graphs for test-dev images**, which we don't):   
    `python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_scene_graph.yaml TEST.SPLIT_VQA testdev_balanced`  
    - generate the submission file on `submission_all` for EvalAI (**This won't work unless you have a file `submission_sceneGraphs.json` under `exp_gqa/gqa_dataset/sceneGraphs/` that contains scene graphs for all images**, which we don't):   
    `python exp_gqa/main.py --cfg exp_gqa/cfgs/lcgn_scene_graph.yaml TEST.SPLIT_VQA submission_all TEST.DUMP_PRED True`

## Acknowledgements

Part of the CLEVR and GQA dataset preprocessing code, many TensorFlow operations (such as `models_clevr/ops.py`) and multi-GPU training code are obtained from the [MAC](https://github.com/stanfordnlp/mac-network) codebase.

The outline of the configuration code (such as `models_clevr/config.py`) is obtained from the [Detectron](https://github.com/facebookresearch/Detectron) codebase.

The ResNet feature extraction script (`exp_clevr/data/extract_features.py`) is obtained from the [CLEVR-IEP](https://github.com/facebookresearch/clevr-iep) codebase.
