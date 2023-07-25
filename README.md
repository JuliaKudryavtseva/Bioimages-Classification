# Bioimages-Classification

Quick start:
---
pip install -r requirements.txt

python class_experiment.py 

# Medical Image Classification with ResNets, EfficientNet and ViT
Medical image classification plays an increasingly important role in healthcare, especially in diagnosing, treatment planning, and disease monitoring. However, the lack of large publicly available datasets with annotations means it is still very difficult, if not impossible, to achieve clinically relevant computer-aided detection and diagnosis (CAD). In recent years, deep learning models have been shown to be very effective at image classification, and they are increasingly being used to improve medical tasks. Thus, this project aims to explore the use of different convolutional neural network (CNN) architectures for medical image classification. Specifically, we will examine the performance of 6 different CNN models (ResNet34, EfficientNet_B0, and ViT) on a dataset of blood cell images.
### Datasets
We will use two datasets for our experiments:
[Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells): This dataset contains 12,500 augmented images of blood cells with 4 different cell types, namely Eosinophil, Lymphocyte, Monocyte, and Neutrophil. 
We use this dataset for the multi-class classification problem.

### Loss function
Multi-class classification refers to the categorization of instances into precisely one class from a set of multiple classes. So, the commonly used loss function is cross-entropy loss. \
### Models
1)	[ResNet](https://arxiv.org/pdf/1512.03385v1.pdf), or Residual Network, is a deep convolutional neural network architecture that was introduced in 2015 by He et al. ResNets work by using residual connections to skip over layers in the network. This allows the network to learn more complex features without becoming too deep and overfitting to the training data.
2)	[EfficientNet](https://arxiv.org/pdf/1905.11946v5.pdf) is a family of convolutional neural network architectures that were introduced in 2019 by Tan et al. EfficientNets are designed to be efficient in terms of both accuracy and computational resources. They achieve this by using a combination of techniques, including compound scaling, squeeze-and-excitation blocks, and autoML.
3)	[ViT](https://arxiv.org/pdf/2010.11929.pdf), or Vision Transformer, is a deep learning model that was introduced in 2020 by Dosovitskiy et al. Vision Transformers are based on the transformer architecture, which was originally developed for natural language processing (NLP).
### Evaluation
![image](https://github.com/JuliaKudryavtseva/Bioimages-Classification/assets/67862423/f7219a46-829a-43eb-898b-8ca80a7fcae7)
![image](https://github.com/JuliaKudryavtseva/Bioimages-Classification/assets/67862423/30ed9c99-7177-4e37-b442-6e16c8f8216d)

<p align="center">
  <img src=![image](https://github.com/JuliaKudryavtseva/Bioimages-Classification/assets/67862423/af6a03dd-32c2-4696-aea0-0a14c585e227)
&nbsp; &nbsp; &nbsp; &nbsp;
 ![image](https://github.com/JuliaKudryavtseva/Bioimages-Classification/assets/67862423/0dbfa032-12a1-40e8-bea5-87cb74e12c88)
</p>

### Conclusion
Unlike others, ViT showed stable learning process and outperformed EfficientNet. The ResNet-34 modification archived the best metrics. However, the difference is not significant, while the number of parameters surpasses by 21 times.
