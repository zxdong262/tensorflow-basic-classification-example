# tensorflow-basic-classification-example
a example of tensorflow keras basic-classification example
based on the tensorflow doc:
- https://www.tensorflow.org/tutorials/keras/basic_classification
- https://www.tensorflow.org/tutorials/keras/basic_text_classification

## run
```bash

# install Helper libraries, for ubuntu 16.04 only
sudo apt-get install python3-pip python3-tk
pip3 install matplotlib numpy scipy matplotlib ipython jupyter pandas sympy nose --user

git clone git@github.com:zxdong262/tensorflow-basic-classification-example.git
cd tensorflow-basic-classification-example
npm i

# for image classification
python3 image.py

# for text classification
python3 text.py

# text classification with weight regularization and dropout
python3 text-adjust.py

# text recognition: random single char image
python3 char.py

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
then use it in `image.py`:
```python
# grab a image to do the test
img1 = 1 - np.array(Image.open('./eg1-sneaker.png')) / 255.0
```

## test your own film comment
just edit text.py
```python
# just copy a negative from imdb
negative = '''<START> This is one of those awful films consisting of too many famous people acting out a plot that has no backbone in the vain hope that their collective fame will patch the holes in the story. I wouldn't wipe my ass with this script.'''

# just copy a positive from imdb
positive = '''<START> I have never seen such an amazing film since I saw The Shawshank Redemption. Shawshank encompasses friendships, hardships, hopes, and dreams. And what is so great about the movie is that it moves you, it gives you hope. Even though the circumstances between the characters and the viewers are quite different, you don't feel that far removed from what the characters are going through.'''
```

## a little more complicated exmaple: captcha reader
https://github.com/zxdong262/tf-captcha-reader