# tensorflow-basic-examples
examples of tensorflow keras
based on the tensorflow doc:
- https://www.tensorflow.org/tutorials/keras

## run
```bash

# install Helper libraries, for ubuntu 16.04 only
## if you miss any libs, just google and install it
sudo apt-get install python3-pip python3-tk
pip3 install matplotlib numpy scipy matplotlib ipython jupyter pandas sympy nose --user


git clone git@github.com:zxdong262/tensorflow-basic-examples.git
cd tensorflow-basic-examples
npm i

# for image classification
python3 examples/image.py

# for text classification
python3 examples/text.py

# text classification with weight regularization and dropout
python3 examples/text-adjust.py

# text recognition: random single char image
python3 examples/char.py

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
img1 = 1 - np.array(Image.open('../data/eg1-sneaker.png')) / 255.0
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