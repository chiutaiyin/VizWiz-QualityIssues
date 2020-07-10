# VizWiz-QualityIssues
Code for the [VizWiz Image Quality Issues Dataset](https://vizwiz.org/tasks-and-datasets/image-quality-issues/), including API, baseline models and evaluation method.

## Requirements ##
- tensorflow v1.14
- keras v2.3.1

## Files ##
```./```
- ```demo_recognizability.ipynb```: demo of recognizability prediction using Detectron or Resnet152 feature maps and its evaluation.
- ```demo_answerability_recognizability.ipynb```: demo of joint prediction of answerability and recognizability using Detectron or Resnet152 feature maps and its evaluation.

```./api```
- ```load_annotations.ipynb```: 
This file shows how to compile [VizWiz-VQA](https://vizwiz.org/tasks-and-datasets/vqa/) and [VizWiz-ImageQualityIssues](https://vizwiz.org/tasks-and-datasets/image-quality-issues/) annotations 
saved in ```./annotations``` into arrays for further use, such as Tensorflow Dataset as in ```./utils/DatasetAPI_to_tensor.py```. Compiled files are saved in ```./data```.


```./annotations```
- ```vqa_annotations/train.json, val.json, test.json``` ([VizWiz-VQA](https://vizwiz.org/tasks-and-datasets/vqa/) training/val/test set)
- ```quality_annotations/train.json, val.json, test.json``` ([VizWiz-ImageQualityIssues](https://vizwiz.org/tasks-and-datasets/image-quality-issues/) training/val/test set)

```./data```
- ```quality.json```: arrayed quality annotations compiled from ```./annotations/quality_annotations/train.json, val.json, test.json``` by following ```./api/load_annotations.ipynb```. This file is for recognizability prediction.
```python 
data = json.load(open('./data/quality.json')) 
# data is a dictionary with keys ['train', 'val', 'test'] corresponding to training/val/test set
# Take training set for example. data['train'] is a dictionary with keys ['image', 'flaws', 'recognizable']
# data['train']['image'], data['train']['flaws'], data['train']['recognizable'] are lists; can be converted numpy array with np.asarray()
# data['train']['image'][0] == the name of first image
# data['train']['flaws'][0] == the flaws of first image
# data['train']['recognizable'][0] == recognizability of first image
```
- ```vqa_quality_merger.json```: arrayed vqa and quality annotations; incorporate ```./annotations/vqa_annotations/train.json, val.json, test.json``` into ```quality.json```.
This file is for joint answerability-recognizability prediction.
```python 
data = json.load(open('./data/vqa_quality_merger.json')) 
# In addition to the keys ['image', 'flaws', 'recognizable'], data['train'] has two other keys ['answerable', 'question']
```

```./fmap```
- ```detectron/```: where image feature maps extracted by [Detectron](https://github.com/facebookresearch/detectron2) are stored. For how to extract the features, please refer to [this notebook](https://colab.research.google.com/drive/1Z9fsh10rFtgWe4uy8nvU4mQmqdokdIRR#scrollTo=UCD0nso8YelA).
- ```resnet152/```: where image feature maps extracted by Resnet152 are stored. They can be derived by following the snippet of code below:
```python
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

resnet152 = keras.applications.ResNet152(include_top=False, weights='imagenet', input_shape=[448, 448, 3])
base_model = keras.models.Model(inputs=resnet152.input, outputs=resnet152.get_layer('conv5_block3_add').output)

img = image.load_img(IMG_PATH, target_size=(448,448)) 
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
img_feat = base_model.predict(img)
```

```./model/VqaQualityModel.py```: Model adapted from [Up-Down attention VQA model](https://arxiv.org/abs/1707.07998) for the joint prediction of answerability and recognizability

```./utils```
- ```DatasetAPI_to_tensor.py```: Tensorflow dataset API for loading data while running the model
- ```word2vocab_vizwiz```: Ids of tokenized frequent words in the questions in VizWiz VQA dataset

```./ckpt_rec``` and ```./ckpt_ans_rec```: checkpoints for the recongizability predictor and Up-Down model for answerability and recognizability prediction, respectively.

## Evaluation results from the baseline models ##
We use average precision as the evaluation metric.

#### Recognizability prediction ####
Avg. precision of **unrecognizability**:

| Feature maps    | Validation set | Test-dev | Test-standard|
| :-------------: |:--------------:| :-------:|:------------:|
| Detectron       | 79.44          | 77.36    | 78.49        |
| Resnet-152      | 80.17          | 77.82    | 78.69        |

#### Answerability and recognizability prediction ####
The format shown below is (avg. precision of **unanswerability** / avg. precision of **unrecognizability given the question is unanswerable**)

| Feature maps    | Validation set | Test-dev      | Test-standard |
| :-------------: |:--------------:| :-----------: |:-------------:|
| Detectron       | 71.41 / 83.08  | 72.26 / 85.38 | 70.53 / 86.20 |
| Resnet-152      | 70.97 / 83.12  | 71.26 / 84.90 | 70.39 / 85.13 |


## References ##
- [VizWiz Project](http://vizwiz.org)
- [Detectron](https://github.com/facebookresearch/detectron2)
- [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)
