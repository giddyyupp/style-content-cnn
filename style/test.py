import torch
from torchvision import transforms
from model_train import ArtModel
# import model_custom
from PIL import Image
import config as cfg
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from sklearn import metrics
import utils
import torch.nn as nn

"""
Testing script. Tests all saved models.
"""
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    img = Image.open(image_path)
    image = img.convert('RGB')
    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def load_dream_model(model_name):
    # Build models
    # ArtModel = model_custom.resnet50(num_classes=3)
    art_model = ArtModel().eval()  # eval mode (batchnorm uses moving mean/variance)
    art_model = art_model.to(device)

    # Load the trained model parameters
    checkpoint = torch.load(cfg.PYTORCH_MODELS + model_name, map_location=lambda storage, loc: storage)
    art_model.load_state_dict(checkpoint['model_state_dict'])

    return art_model


def main():

    THRESHOLD = 0.0

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((cfg.RESIZE, cfg.RESIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    test_files, test_labels = utils.prep_data(cfg.test_path)
    # test_files, test_labels = load_data_set_test(test_path)
    test_labels = list(map(int, test_labels))

    for i in range(399, 4, -5):
        test_res = []

        model_name = "art_model-{}.pkl".format(i)
        dream_model = load_dream_model(model_name)
        print "Model Name: " + model_name

        for im_name in test_files:
            # Prepare an image
            image = load_image(im_name, transform)
            image_tensor = image.to(device)
            # test image
            predicted_label = dream_model(image_tensor) # convert to numpy array
            m = nn.Softmax()
            norm_labels = m(predicted_label)

            norm_labels_np = norm_labels.cpu().data.numpy()
            predicted_label = np.argmax(norm_labels_np)
            if norm_labels_np[0][predicted_label] >= THRESHOLD:
                test_res.append(predicted_label)
            else:
                test_res.append(-1)

            # print "Image Name: " + im_name + " Test Res: " + cfg.CLASS_LIST[predicted_label] + " Confidence: " + str(norm_labels_np[0][predicted_label])

        score = metrics.accuracy_score(test_labels, test_res)
        cls_report = metrics.classification_report(test_labels, test_res)
        conf_mat = metrics.confusion_matrix(test_labels, test_res)

        print "Accuracy = " + str(score)
        print cls_report
        print conf_mat


if __name__ == '__main__':
    main()
