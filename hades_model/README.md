## Hades Model

### Dataset - HARRISON

- maximum numbers of hashtag for one image is 10.
- minimum numbers of hashtag for one image is 1.
- 57,383 images, 2/3 for training 1/3 for testing.
- 50 classes

### Feature extraction

Use ImageNet pretrained model resnet50 to extract image feature as input data for non-end2end model.

	$ python image_extractor.py

### Word2vec

Use skip-gram algorithm to build our word2vec model.

	$ python word2vec.py

||| 
|parameters  |default|
|-m |select mode, train or test, default is train|
|-bs |  batch size, default is 64| 
|-epos | numbers of epochs, default is 100|
|-o | path of output file, default is ./|
|-lr |learning rate, default is 1e-4|
|-th |threshold of counting number of word, default is 4 |
|-w |window size of skip-gram, default is 3|
|-embed |dimension of embedding vector, default is 500|

### How to run our main program

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


