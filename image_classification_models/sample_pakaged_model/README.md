# sample_pakage_model

This is code to sample pakaged model of classify images.

<b>Folder sturacture</b>
```
├── LICENSE.txt
├── README.md
├── models
│   ├── __init__.py
│   ├── cnn_model.py
│   ├── pyramid_pooling_module.py
│   └── scse.py
├── requirements.txt
├── sample_pakage_model
│   ├── __init__.py
│   ├── evaluate.py
│   └── train.py
└── setup.py
```



# To prepare package environment

```
# create virtual environment
Python3.7 -m venv venv && source venv/bin/activate

# install modules
pip install -r requirements.txt

# check whether modules in requirements.txt were installed 
pip freeze  

# make package to provide
python setup.py sdist
```


# Train command
When training, write train image and evaluation image as follows:

```
python robotics_test/train.py --class_a_npy 'class_a.npy' --class_b_npy 'class_b.npy' --field_npy 'field.npy'
```


# What I did and attention points

1 ```brew install tree```

2.checked tensorflow and Keras version at google colab and wrote its version to requirements.txt
```
Import keras
print(keras.__version__)
```

3.「np.load error」 does not occur
After install requirements.txt, run ```python train.py```command. Then 「np.load error」 does not occur.

4.reference site:

https://qiita.com/Kensuke-Mitsuzawa/items/7717f823df5a30c27077

https://github.com/fizyr/keras-maskrcnn


5.Use serious arg(parser.add_argument)

6.To know more detail, you should read your own book about python pakeging.
