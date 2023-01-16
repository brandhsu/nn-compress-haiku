# Neural Network Compression - [Haiku](https://github.com/deepmind/dm-haiku)

Implementation of several neural network compression techniques (pruning, quantization, factorization), in Haiku.

<div align='center'>
  <kbd>
    <a href='http://graduatestudent.ucmerced.edu/yidelbayev/papers/cikm21/cikm21_slides.pdf'>
      <img src='figs/yidelbayev-cikm21_slides.png' height=480/>
    </a>
  </kbd>
</div>

For an introduction to neural network compression, see [4-popular-model-compression-techniques-explained](https://xailient.com/blog/4-popular-model-compression-techniques-explained).

## Installation

```shell
$ git clone https://github.com/Brandhsu/nn-compress-haiku
$ pip install -r nn-compress-haiku/requirements.txt
```

## Usage

First, train a model on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

```shell
$ python scripts/01_train.py --save-dir models
```

Then compress it!

```shell
$ python scripts/02_compress.py --model-path models/params.pkl --compression-func svd --save-dir figs
```

> Note: Compression happens post-training in a layer-by-layer (local) fashion.

## Results

| Type                                       | Accuracy | Latency |
| ------------------------------------------ | -------- | ------- |
| [pruning](nn_compress_haiku/prune.py)      | Accuracy | Latency |
| [quantization](nn_compress_haiku/quant.py) | Accuracy | Latency |
| [factorization](nn_compress_haiku/svd.py)  | Accuracy | Latency |

> Note: The results shown are attained with the default settings.

Remarks:

- Accuracy decreases with compression, and seems to perform best on the CIFAR-10 test set.
- Latency does not decrease with compression since the number of matrix multiplication operations remain the same.
