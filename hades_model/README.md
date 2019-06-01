## Hades Model

### Dataset - HARRISON

- maximum numbers of hashtag for one image is 10.
- minimum numbers of hashtag for one image is 1.
- 57,383 images, 2/3 for training 1/3 for testing.
- 

### Feature extraction

- Use ImageNet pretrained model to extract image feature as input data for non-end2end model.

### Word2vec

- 

### How to run our programs

    $ python main.py -m <mode> -model_type <model_type> -end2end <end2end>

||| 
|---|---|
|parameters  |default|
|-m|select mode, train or test, default is train|
|-bs|  batch size, default is 64| 
|-epos| numbers of epochs, default is 100|
|-s| path of saving model, default is model/|
|-lr|learning rate, default is 1e-4|
|-model_type|four types can be choose, [dnn, multi_label, devise, heracles]|
|-end2end|training end2end or not, default is False|
||| 


