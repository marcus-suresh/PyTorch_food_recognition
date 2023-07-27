# -*- coding: utf-8 -*-
"""

# PyTorch object detection model training

[PyTorch](https://pytorch.org/) datasets provide a great starting point for loading complex datasets, letting you define a class to load individual samples from disk and then creating data loaders to efficiently supply the data to your model. Problems arise when you want to start iterating over your dataset itself. PyTorch datasets are fairly rigid and require you to either rewrite them or the underlying data on disk if you want to make any changes to the data you are training or testing your model on. That is where [FiftyOne](http://fiftyone.ai) comes in.

PyTorch datasets can synergize well with [FiftyOne datasets](https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#using-fiftyone-datasets) for hard computer vision problems like [classification, object detection, segmentation, and more](https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#labels). 
The flexibility of FiftyOne datasets lets you easily experiment with and finetune the datasets you use for training and testing to create better-performing models, faster.
In this example, I am focusing on object detection since that is one of the most common vision tasks while also being fairly complex. However, these methods work for most machine learning tasks. Specifically, this notebook covers:
* Loading your labeled dataset into FiftyOne
* Writing a PyTorch object detection dataset that utilizes your loaded FiftyOne dataset
* Exploring views into your FiftyOne dataset for training and evaluation
* Training a Torchvision object detection model on your FiftyOne dataset views
* Evaluating your models in FiftyOne to refine your dataset

## Setup

**Before you begin!** If you are running this in Colab, make sure you select a GPU instance under `Runtime` > `Change runtime type` since this notebook contains some model training.

To start, we need to install fiftyone, pytorch, and torchvision, as well as clone the torchvision GitHub repository to use the training and evaluation utilities provided for the [Torchvision Object Deteciton Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset) that we are using to train a basic object detection model.
"""

!pip install fiftyone
!pip install torch torchvision

"""If you run into a `cv2` error when importing FiftyOne later on, it is an issue with OpenCV in Colab environments. [Follow these instructions to resolve it.](https://github.com/voxel51/fiftyone/issues/1494#issuecomment-1003148448)"""

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# 
# # Download TorchVision repo to use some files from
# # references/detection
# git clone https://github.com/pytorch/vision.git
# cd vision
# git checkout v0.3.0
# 
# cp references/detection/utils.py ../
# cp references/detection/transforms.py ../
# cp references/detection/coco_eval.py ../
# cp references/detection/engine.py ../
# cp references/detection/coco_utils.py ../
#

"""## Loading your data

Getting your data into FiftyOne is oftentimes actually easier than getting it into a PyTorch dataset. Additionally, once the data is in FiftyOne it is much more flexible allowing you to easily find and access even the most specific subsets of data that you can then use to train or evaluate your model.

If you have data that follows a certain format on disk ([for example a directory tree for classification, the COCO detection format, or many more](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/index.html#id1)), then you can load it into FiftyOne in [one line of code](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/index.html).

In this notebook, I am going to work with [COCO-2017](https://cocodataset.org/#home) and load it from the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/index.html).
"""

import torch

torch.manual_seed(1)

import fiftyone as fo
import fiftyone.zoo as foz

fo_dataset = foz.load_zoo_dataset("coco-2017", "validation")

"""We will be needing the height and width of images later in this notebook so we need to compute metadata on our dataset."""

fo_dataset.compute_metadata()

"""We can create a session and visualize this dataset in the [FiftyOne App](https://voxel51.com/docs/fiftyone/user_guide/app.html)."""

session = fo.launch_app(fo_dataset)

"""## PyTorch dataset and training setup

A [PyTorch dataset](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class) is a class that defines how to load a static dataset and its labels from disk via a simple iterator interface. They differ from FiftyOne datasets which are flexible representations of your data geared towards visualization, querying, and understanding.

Every PyTorch model expects data and labels to pass into it in a certain format. Before being able to write up a PyTorch dataset class, you first need to understand the format that the model requires. Namely, we need to know exactly what format the data loader is expected to output when iterating through the dataset so that we can properly define the `__getitem__` method in the PyTorch dataset.
In this example, I am following the [Torchvision object detection tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset) and construct a PyTorch dataset to work with their RCNN-based models.
"""

import torch
import fiftyone.utils.coco as fouc
from PIL import Image


class FiftyOneTorchDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.
    
    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for training or testing
        transforms (None): a list of PyTorch transforms to apply to images and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset that contains the 
            desired labels to load
        classes (None): a list of class strings that are used to define the mapping between
            class names and indices. If None, it will use all classes present in the given fiftyone_dataset.
    """

    def __init__(
        self,
        fiftyone_dataset,
        transforms=None,
        gt_field="ground_truth",
        classes=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(
                "%s.detections.label" % gt_field
            )

        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        area = []
        iscrowd = []
        detections = sample[self.gt_field].detections
        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_label(
                det, metadata, category_id=category_id,
            )
            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes

"""The following code loads Faster-RCNN with a ResNet50 backbone from Torchvision and modifies the classifier for the number of classes we are training on:"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

"""For this example, we are going to write a simple training loop. This function is going to take a model and our PyTorch datasets as input and use the [`train_one_epoch()`](https://github.com/pytorch/vision/blob/master/references/detection/engine.py) and [`evaluate()`](https://github.com/pytorch/vision/blob/master/references/detection/engine.py) functions from the Torchvision object detection code"""

# Import functions from the torchvision references we cloned
from engine import train_one_epoch, evaluate
import utils

def do_training(model, torch_dataset, torch_dataset_test, num_epochs=4):
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        torch_dataset, batch_size=2, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        torch_dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device %s" % device)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

"""## FiftyOne views and datasets

One of the primary ways of interacting with your FiftyOne dataset is through different [views](https://voxel51.com/docs/fiftyone/user_guide/using_views.html) into your dataset. These are constructed by applying operations like filtering, sorting, slicing, etc, that result in a specific view into certain labels/samples of your dataset. These operations make it easier to experiment with different subsets of data and continue to finetune your dataset to train better models.

For example, cluttered images make it difficult for models to localize objects. We can use FiftyOne to create a view containing only samples with more than, say, 10 objects. You can perform the same operations on views as datasets, so we can create an instance of our PyTorch dataset from this view:
"""

from fiftyone import ViewField as F

busy_view = fo_dataset.match(F("ground_truth.detections").length() > 10)

busy_torch_dataset = FiftyOneTorchDataset(busy_view)

session.view = busy_view

"""Another example is if we want to train a model that is used primarily for road vehicle detection. We can easily create training and testing views (and corresponding PyTorch datasets) that only contain the classes car, truck, and bus:"""

from fiftyone import ViewField as F

vehicles_list = ["car", "truck", "bus"]
vehicles_view = fo_dataset.filter_labels("ground_truth",
        F("label").is_in(vehicles_list))

print(len(vehicles_view))

session.view = vehicles_view

# From the torchvision references we cloned
import transforms as T

train_transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5)])
test_transforms = T.Compose([T.ToTensor()])

# split the dataset in train and test set
train_view = vehicles_view.take(500, seed=51)
test_view = vehicles_view.exclude([s.id for s in train_view])

# use our dataset and defined transformations
torch_dataset = FiftyOneTorchDataset(train_view, train_transforms,
        classes=vehicles_list)
torch_dataset_test = FiftyOneTorchDataset(test_view, test_transforms, 
        classes=vehicles_list)

"""## Training and Evaluation

In this section, we use the functions and datasets we defined above to initialize, train, and evaluate a model continuing with the vehicle example. 
"""

model = get_model(len(vehicles_list)+1)

do_training(model, torch_dataset, torch_dataset_test, num_epochs=4)

"""One of the main draws of FiftyOne is the ability to find failure modes of your model. The [built-in evaluation protocols](https://voxel51.com/docs/fiftyone/user_guide/evaluation.html#evaluating-models) will help you find where your model got things right and where it got things wrong. Before we can evaluate the model, we need to run it on our test set and store the results in FiftyOne. Doing this is fairly simple and just requires us to run inference for the test images, get their corresponding [FiftyOne samples](https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#samples), and add a new field called `predictions` to each sample to store the detections."""

import fiftyone as fo

def convert_torch_predictions(preds, det_id, s_id, w, h, classes):
    # Convert the outputs of the torch model into a FiftyOne Detections object
    dets = []
    for bbox, label, score in zip(
        preds["boxes"].cpu().detach().numpy(), 
        preds["labels"].cpu().detach().numpy(), 
        preds["scores"].cpu().detach().numpy()
    ):
        # Parse prediction into FiftyOne Detection object
        x0,y0,x1,y1 = bbox
        coco_obj = fouc.COCOObject(det_id, s_id, int(label), [x0, y0, x1-x0, y1-y0])
        det = coco_obj.to_detection((w,h), classes)
        det["confidence"] = float(score)
        dets.append(det)
        det_id += 1
        
    detections = fo.Detections(detections=dets)
        
    return detections, det_id

def add_detections(model, torch_dataset, view, field_name="predictions"):
    # Run inference on a dataset and add results to FiftyOne
    torch.set_num_threads(1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device %s" % device)

    model.eval()
    model.to(device)
    image_paths = torch_dataset.img_paths
    classes = torch_dataset.classes
    det_id = 0
    
    with fo.ProgressBar() as pb:
        for img, targets in pb(torch_dataset):
            # Get FiftyOne sample indexed by unique image filepath
            img_id = int(targets["image_id"][0])
            img_path = image_paths[img_id]
            sample = view[img_path]
            s_id = sample.id
            w = sample.metadata["width"]
            h = sample.metadata["height"]
            
            # Inference
            preds = model(img.unsqueeze(0).to(device))[0]
            
            detections, det_id = convert_torch_predictions(
                preds, 
                det_id, 
                s_id, 
                w, 
                h, 
                classes,
            )
            
            sample[field_name] = detections
            sample.save()

add_detections(model, torch_dataset_test, fo_dataset, field_name="predictions")

results = fo.evaluate_detections(
    test_view, 
    "predictions", 
    classes=["car", "bus", "truck"], 
    eval_key="eval", 
    compute_mAP=True
)

"""The [DetectionResults](https://voxel51.com/docs/fiftyone/api/fiftyone.utils.eval.detection.html#fiftyone.utils.eval.detection.DetectionResults) object that is returned stores information like the mAP and contains functions that let you [plot confusion matrices, precision-recall curves, and more](https://voxel51.com/docs/fiftyone/user_guide/evaluation.html). Also, these evaluation runs are tracked in FiftyOne and can be managed through functions like [list_evaluations()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.list_evaluations)."""

results.mAP()

results.print_report()

"""By default, objects are only matched with other objects of the same class. In order to get an interesting confusion matrix, we need to match interclass objects by setting `classwise=False`."""

results_interclass = fo.evaluate_detections(
    test_view, 
    "predictions", 
    classes=["car", "bus", "truck"], 
    compute_mAP=True, 
    classwise=False
)

results_interclass.plot_confusion_matrix()

"""Note that there appears to be confusion between car and truck classes.

The [detection evaluation](https://voxel51.com/docs/fiftyone/user_guide/evaluation.html#detections) also added the attributes `eval_fp`, `eval_tp`, and `eval_fn` to every predicted detection indicating if it is a false positive, true positive, or false negative. 
Let's create a view to find the worst samples by sorting by `eval_fp` using the [FiftyOne App](https://voxel51.com/docs/fiftyone/user_guide/app.html) to visualize the results.
"""

session.view = test_view.sort_by("eval_fp", reverse=True)

"""Looking through some of these samples, we can see the confusion between the classes "car" and "truck" that we found earlier. However, this seems to be, at least in part, due to annotation errors on vans and SUVs where they are interchangably labeled as "car" and "truck". The example below shows an SUV annotated as "truck" but predicted as "car"."""

session.view = test_view.sort_by("eval_fp", reverse=True)

"""It would be best to get this [data reannotated to fix these mistakes](https://towardsdatascience.com/managing-annotation-mistakes-with-fiftyone-and-labelbox-fc6e87b51102), but in the meantime, we can easily remedy this by simply creating a new view that remaps the labels `car`, `truck`, and `bus` all to `vehicle` and then retraining the model with that. This is only possible because we are backing our data in FiftyOne and loading views into PyTorch as needed. Without FiftyOne, the PyTorch dataset class or the underlying data would need to be changed to remap these classes."""

# map labels to single vehicle class 
vehicles_map = {c: "vehicle" for c in vehicles_list}

train_map_view = train_view.map_labels("ground_truth", vehicles_map)
test_map_view = test_view.map_labels("ground_truth", vehicles_map)

# use our dataset and defined transformations
torch_map_dataset = FiftyOneTorchDataset(train_map_view, train_transforms)
torch_map_dataset_test = FiftyOneTorchDataset(test_map_view, test_transforms)

# Only 2 classes (background and vehicle)
vehicle_model = get_model(2)

do_training(vehicle_model, torch_map_dataset, torch_map_dataset_test)

add_detections(vehicle_model, torch_map_dataset_test, test_map_view, field_name="vehicle_predictions")

vehicle_results = fo.evaluate_detections(
    test_map_view, 
    "vehicle_predictions", 
    classes=["vehicle"], 
    eval_key="vehicle_eval", 
    compute_mAP=True
)

vehicle_results.mAP()

vehicle_results.print_report()

"""Due to our ability to easily visualize and manage our dataset with FiftyOne, we were able to spot and take action on a dataset issue that would otherwise have gone unnoticed if we only concerned ourselves with dataset-wide evaluation metrics and fixed dataset representations. Through these efforts, we managed to increase the mAP of the model to 43%.

Even though this example workflow may not work in all situations, this kind of class-merging strategy can be effective in cases where more fine-grained discrimination is not called for.

## Summary

PyTorch and related frameworks provide quick and easy methods to bootstrap your model development and training pipelines. However, they largely overlook the need to massage and finetune datasets to efficiently improve performance. FiftyOne makes it easy to load your datasets into a flexible format that works well with existing tools allowing you to provide better data for training and testing. As they say, "garbage in, garbage out".
"""
