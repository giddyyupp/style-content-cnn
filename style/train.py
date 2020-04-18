import torch
import torch.nn as nn
import numpy as np
import os
from model_train import ArtModel
# import model_custom
from torch.autograd import Variable
from torchvision import transforms
from data_loader import get_loader
import datetime
# import n_random_crops

import config as cfg
import utils

"""
Train script. 
Read config.py for details
"""


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def main():
    # Create model directory
    if not os.path.exists(cfg.PYTORCH_MODELS):
        os.makedirs(cfg.PYTORCH_MODELS)

    # open loss info
    today = datetime.datetime.now()
    loss_info = open(cfg.PYTORCH_MODELS + 'loss_' + str(today) + '.txt', 'w')

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(cfg.PATCH_SIZE),
        # n_random_crops.NRandomCrop(cfg.PATCH_SIZE, 10),
        # transforms.Resize((cfg.RESIZE, cfg.RESIZE)), # avgpool u 3x3 yapinca 96 olmali.
        transforms.Pad((224-cfg.PATCH_SIZE)/2, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Build the models
    art_model = ArtModel()
    # art_model = model_custom.resnet50(pretrained=True, num_classes=cfg.CLASS_COUNT)
    if torch.cuda.is_available():
        art_model.cuda()

    if cfg.LOAD_TRAINED_MODEL:
        art_model.load_state_dict(torch.load(cfg.PYTORCH_MODELS + cfg.MODEL_NAME))

    # load dataset and labels
    # image_path_list, labels_list = load_data_set()
    image_path_list, labels_list = utils.prep_data(cfg.train_path)

    # append main path
    # image_path_list = [image_main_path + imm for imm in image_path_list]

    # Build data loader
    data_loader = get_loader(image_path_list, labels_list, cfg.BATCH_SIZE, shuffle=True, transform=transform, num_workers=cfg.NUM_WORKERS)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, list(art_model.parameters())), lr=cfg.LEARNING_RATE)

    # Train the Models
    total_step = len(data_loader)
    for epoch in range(1, cfg.EPOCH_COUNT):
        for i, (images, label) in enumerate(data_loader):
            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            # Set mini-batch ground truth
            label = to_var(label, volatile=False)
            # Forward, Backward and Optimize
            art_model.zero_grad()
            # feed images to CNN model
            predicted_label = art_model(images)

            loss = criterion(predicted_label, label)
            loss.backward()
            optimizer.step()

            # Print log info
            if i % cfg.LOG_STEP == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.7f'
                      % (epoch, cfg.EPOCH_COUNT, i, total_step, loss))

                loss_info.write('Epoch [%d/%d], Step [%d/%d], Loss: %.7f\n'
                                % (epoch, cfg.EPOCH_COUNT, i, total_step, loss))

        # Save the models
        if (epoch + 1) % cfg.SAVE_PERIOD_IN_EPOCHS == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': art_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(cfg.PYTORCH_MODELS, 'art_model-%d.pkl' % epoch))
            # torch.save(art_model.state_dict(),
            #            os.path.join(cfg.PYTORCH_MODELS,
            #                         'art_model-%d.pkl' % epoch))


if __name__ == '__main__':
    main()
