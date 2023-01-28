
<a name="readme-top"></a>



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="">
    <img src="images/increase.png" alt="Logo" width="160" height="160">
  </a>

  <h1 align="center">MegaBoost</h1>

  <p align="center">
    Semi-supervised AutoML library on top of PyTorch.
    <br />
    <!-- <a href=""><strong>Explore the docs »</strong></a> -->
    <br />
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

 A state-of-the-art semi-supervised AutoML library on top of Pytorch to minimize your time and learning curve in machine learning. It is built for: 

 1. Semi-supervised model training with unlabeled data and very limited labeled data. 

 2. Fine tuning deep learning models.

 3. Boosting model performance.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Installation
Make sure you have installed Pytorch and Torchvision from the [official](https://pytorch.org/) site. Then you can simply install this library from PyPI: 

  ```sh
    pip install megaboost
  ```

<!-- GETTING STARTED -->
## Getting Started
Import the libraries:
  ```sh
    import torch
    import megaboost as mg

    labeled_dataset, unlabeled_dataset, test_dataset = mg.prepare_cifar10(resize=RESIZE)

    megaboost = mg.MegaBoost(config=config)
  ```

  Train the model using a similar style in scikit-learn:


  ```sh
    megaboost.fit(labeled_loader, test_loader, unlabeled_loader)
  ```

  Use the model:

  ```sh
    res = megaboost.predict(image)
  ```


You can find the colab demo [here](https://colab.research.google.com/drive/1SKVzkZGFdtZ8uJz-ubSa3aTf5lWCqBcM?usp=sharing). 

[MegaBoost Tutorial 1: Fine-tune Image Classification Model](https://colab.research.google.com/drive/1tVL_Z6Dsi9lNOCanRcuXCboyrdTS2q9w?usp=sharing)

<!-- ROADMAP -->
## Roadmap

- [x] Enable MPS acceleration on Mac
- [x] Enable automatic mixed precision by default
- [x] SSL: image classification
- [ ] 

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- CONTACT -->
## Contact

 maxnghello at gmail.com

Follow me on Medium: [@medium.data.scientist](https://medium.com/@data.scientist)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## More

MegaBoost is an ensemble of state-of-the-art SSL methods with [Self Meta Pseudo Labels](https://arxiv.org/abs/2212.13420) as the backbone.






