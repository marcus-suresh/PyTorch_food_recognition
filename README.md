# Food Ingredients Recognition through Computer Vision Multi-Label Learning

This repository contains the code for applying a PyTorch-centric CV-based ingredients recognition application through multi-label learning.

![alt text](https://github.com/marcus-suresh/PyTorch_food_recognition/blob/main/PyTorch.png)

## Abstract
The proliferation of food-related images dominating social media is a reflection of modern culture. Food has transcended from being a form of mere nourishment, to what can now be characterised as an essential component for human interaction and bonding. It is therefore unsurprising that food-related images are a topic of keen interest. In this paper, we present our computer vision application that identifies ingredients visible in a dish and retrieves possible recipes using those ingredients to encourage culinary inquisity. We combine *computer vision, deep learning and natural language processing* to present the end-user with curated intelligence to explore and create new dishes using a variety of recipes. In this paper, we outline our methodology and performance results in detecting and retrieving the necessary information using multimedia data. Overall, our Food Ingredient Retrieval application achieves a Mean Average Precision of 0.44.


## Pipeline
In order to achieve our aim of retrieving gourmet recipes from a single image of food, we must first expose the Regional-Based Convolutional Neural Network to the training data of food images along with their labels. Ultimately this will help us achieve our goal. That is an application that accepts as an input - a single food which the RCNN algorithm will process and analyse as the source target for retrieving recipes.  

## Methodology
The methodology that we have adopted is outlined below:
* Firstly, we set up our Google Colab environment by mounting to Google Drive, and the import the necessary Python 3 packages along with their dependencies which include **Pytorch, Torchvision, OpenCV and FiftyOne**.
* We then ingest the **Food 101 data set** and iterate through the images in the output folder including all its subdirectories and create a new directory to store all images without hierarchy sub-folders on to our mounted Google Drive. The length of the directory now stands at 101,000 images. Given the sheer size of the dataset and our limited computational resources, we took an approximate 5 per cent sub-sample of the data of 5,000 randomised images. As we have unlabeled images, we will first need to generate object labels. We can generate ground truth labels with an existing pretrained model.
* We then load unlabeled data into FiftyOne (an open-source tool for building high-quality datasets and computer vision models) and generate predictions with the FiftyOne Model Zoo, which is contains a collection of pre-trained models that can be downloaded and run inference on any FiftyOne data sets.
* Using the labelled data from the previous step, we then export the labelled data into COCO format which is a large image data set designed for object detection, segmentation and caption generation. The rationale for using the COCO format is due to the superior performance. It contains 2.5 million labelled instances in 382,000 images. Given this, the literature suggests that the COCO dataset is particularly adept at training models to detect objects such as in our case. 
* Next we deploy PyTorch and load a model that has been pre-trained using the COCO data. To further improve our model, we limit the ingredients to a small subset bringing it down from 5,000 to 3108 images. We then apply a split/train of 2,500 images for training and test on 608 images. 
* Once the ingredients have been identified, we take those identified ingredients, and deploy **natural language processing** and **web-browsing** to search online websites for potential cooking receipes.

## Results
The application is able to detect objects including food ingredients from the target as shown below. 
![alt text](https://github.com/marcus-suresh/PyTorch_food_recognition/blob/main/11.png)
![alt text](https://github.com/marcus-suresh/PyTorch_food_recognition/blob/main/12.png)

## Citation
M.Suresh (2022)
https://github.com/marcus-suresh

