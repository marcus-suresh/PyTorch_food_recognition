# Food Ingredients Recognition through Computer Vision Multi-Label Learning

This repository contains the code for applying a PyTorch-centric CV-based ingredients recognition application through multi-label learning.

## Abstract
The proliferation of food-related images dominating social media is a reflection of modern culture. Food has transcended from being a form of mere nourishment, to what can now be characterised as an essential component for human interaction and bonding. It is therefore unsurprising that food-related images are a topic of keen interest. In this paper, we present our computer vision application that identifies ingredients visible in a dish and retrieves possible recipes using those ingredients to encourage culinary inquisity. We combine *computer vision, deep learning and natural language processing* to present the end-user with curated intelligence to explore and create new dishes using a variety of recipes. In this paper, we outline our methodology and performance results in detecting and retrieving the necessary information using multimedia data. Overall, our Food Ingredient Retrieval application achieves a Mean Average Precision of 0.44.


## Pipeline
In order to achieve our aim of retrieving gourmet recipes from a single image of food, we must first expose the Regional-Based Convolutional Neural Network to the training data of food images along with their labels. Ultimately this will help us achieve our goal. That is an application that accepts as an input - a single food which the RCNN algorithm will process and analyse as the source target for retrieving recipes.  


## Datasets

The datasets Ingredients101 and Recipes5k were used to evaluate this model:
* [Ingredients 101](http://www.ub.edu/cvub/ingredients101/):  It consists of the list of most common ingredients for each of the 101 types of food contained in the Food101 dataset, making a total of 446 unique ingredients (9 per recipe on average). The dataset was divided in training, validation and test splits making sure that the 101 food types were balanced. We make public the lists of ingredients together with the train/val/test split applied to the images from the Food101 dataset.
* [Recipes5k](http://www.ub.edu/cvub/recipes5k/): Dataset for ingredients recognition with 4,826 unique recipes composed of an image and the corresponding list of ingredients. It contains a total of 3,213 unique ingredients (10 per recipe on average) and a simplified version of 1,013 ingredients. Each recipe is an alternative way to prepare one of the 101 food types in Food101. Hence, it captures at the same time the intra-class variability and inter-class similarity of cooking recipes. The nearly 50 alternative recipes belonging to each of the 101 classes were divided in train, val and test splits in a balanced way. We make also public this dataset together with the splits division.

## Citation
