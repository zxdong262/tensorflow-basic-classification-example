# tensorflow-basic-classification-example
a example of tensorflow keras basic-classification example
based on the tensorflow doc: https://www.tensorflow.org/tutorials/keras/basic_classification

## run
```bash

# install Helper libraries, for ubuntu 16.04 only
sudo apt-get install python3-pip python3-tk
pip3 install matplotlib numpy scipy matplotlib ipython jupyter pandas sympy nose --user

git clone git@github.com:zxdong262/tensorflow-basic-classification-example.git
cd tensorflow-basic-classification-example
python3 main.py

## if you miss any libs, just google and install it
```


## test your own test image
just create/search a 28*28 greyscale image in one of these categories:
```python
[
  'T-shirt/top',
  'Trouser',
  'Pullover',
  'Dress',
  'Coat',
  'Sandal',
  'Shirt',
  'Sneaker',
  'Bag',
  'Ankle boot'
]
```
then use it in `main.py`:
```python
# grab a image to do the test
img = misc.imread('./eg1-sneaker.png', flatten=True)
```
