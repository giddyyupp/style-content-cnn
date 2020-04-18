import torch
from torchvision import transforms
from model_train import ContentModel
from PIL import Image
import config as cfg
import numpy as np
from sklearn import metrics
import utils
import torch.nn as nn

"""
Batch test script.
"""
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    img = Image.open(image_path)
    image = img.convert('RGB')
    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def load_dream_model():
    # Build models
    content_model = ContentModel().eval()  # eval mode (batchnorm uses moving mean/variance)
    content_model = content_model.to(device)

    # Load the trained model parameters
    checkpoint = torch.load(cfg.PYTORCH_MODELS + cfg.MODEL_NAME, map_location=lambda storage, loc: storage)
    content_model.load_state_dict(checkpoint['model_state_dict'])

    return content_model


def load_data_set_test(test_path):
    # image path lerinin oldugu dosya
    with open(test_path + "file_names_test.txt") as f:
        image_path_list = f.read().splitlines()
    # image lar ile ayni sirada olacak sekilde label bilgisi
    with open(test_path + "labels_test.txt") as f:
        labels_list = f.read().splitlines()

    return image_path_list, labels_list


def main():

    THRESHOLD = 0.0

    dream_model = load_dream_model()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    test_files, test_labels = utils.prep_data(cfg.test_path)
    # test_files, test_labels = load_data_set_test(cfg.test_path)
    test_labels = list(map(int, test_labels))

    test_res = []

    for im_name in test_files:
        # Prepare an image
        image = load_image(im_name, transform)
        image_tensor = image.to(device)
        # test image
        predicted_label = dream_model(image_tensor)
        m = nn.Softmax()
        norm_labels = m(predicted_label)

        norm_labels_np = norm_labels.cpu().data.numpy() # convert to numpy array
        predicted_label = np.argmax(norm_labels_np)
        if norm_labels_np[0][predicted_label] >= THRESHOLD:
            test_res.append(predicted_label)
        else:
            test_res.append(-1)

        print "Image Name: " + im_name + " Test Res: " + cfg.CLASS_LIST[predicted_label] + " Confidence: " + str(norm_labels_np[0][predicted_label])

    # calculate metrics
    score = metrics.accuracy_score(test_labels, test_res)
    cls_report = metrics.classification_report(test_labels, test_res)
    conf_mat = metrics.confusion_matrix(test_labels, test_res)

    print "Accuracy = " + str(score)
    print cls_report
    print conf_mat


if __name__ == '__main__':
    main()
