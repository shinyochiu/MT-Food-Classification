import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
import numpy as np
import csv
import data_reorganization
from models import MTFoodClassify, MTFoodFeature

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Food Image Retrieval Training')
    parser.add_argument('--architecture', type=str, default='resnet101')
    parser.add_argument("--data_dir", type=str, default="../data/",
                        help="directory in which training data ")
    parser.add_argument("--test_data_dir", type=str, default='',
                        help="directory in which training data ")
    parser.add_argument("--encoder_dir", type=str, default="../encoder/", help="directory in which pretrained feature extract model should be saved")
    parser.add_argument("--model_dir", type=str, default="../model/", help="directory in which training state and model should be saved")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of training epochs")
    parser.add_argument("--num_classes", type=int, default=1000, help="number of data classes")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--input_size", type=int, default=2048, help="number of inputs to the classifier")
    parser.add_argument("--layer1_size", type=int, default=64, help="number of units in the classifier")

    return parser.parse_args()

arglist = parse_args()

def extract_feat(img_encoder, inputs):
    img_encoder.eval()
    inputs = inputs.to(img_encoder.device)
    outputs = img_encoder(inputs)
    return outputs

def train_model(img_encoder, food_classifier, train_loader, valid_loader, criterion, num_epochs=20):
    from tqdm import tqdm

    def train(food_classifier, train_loader, criterion):
        total_loss = 0.0
        total_correct = 0
        food_classifier.train()
        for inputs, labels, _ in tqdm(train_loader, desc="train"):
            inputs = inputs.to(food_classifier.device)
            labels = labels.to(food_classifier.device)
            food_classifier.optimizer.zero_grad()
            #features = extract_feat(img_encoder, inputs)
            #features = torch.reshape(features, (-1, arglist.input_size))
            outputs = food_classifier(inputs)
            loss = criterion(outputs, labels)
            pred = torch.softmax(outputs, dim=-1)
            _, predictions = torch.topk(pred, 3)
            loss.backward()
            food_classifier.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            label = torch.reshape(labels.data, (labels.data.size()[0], 1))
            total_correct += torch.sum(label == predictions)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(food_classifier, valid_loader, criterion):
        total_loss = 0.0
        total_correct = 0
        food_classifier.eval()
        for inputs, labels, _ in tqdm(valid_loader, desc="valid"):
            inputs = inputs.to(food_classifier.device)
            labels = labels.to(food_classifier.device)
            food_classifier.optimizer.zero_grad()
            #features = extract_feat(img_encoder, inputs)
            #features = torch.reshape(features, (-1, arglist.input_size))
            outputs = food_classifier(inputs)
            loss = criterion(outputs, labels)
            pred = torch.softmax(outputs, dim=-1)
            _, predictions = torch.topk(pred, 3)
            total_loss += loss.item() * inputs.size(0)
            label = torch.reshape(labels.data, (labels.data.size()[0], 1))
            total_correct += torch.sum(label == predictions)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    for epoch in tqdm(range(num_epochs), desc="epoch", position=0, leave=True):
        #print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        #print('*' * 100)
        train_loss, train_acc = train(food_classifier, train_loader, criterion)
        tqdm.write("training loss: {:.4f}, acc: {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(food_classifier, valid_loader, criterion)
        tqdm.write("validation loss: {:.4f}, acc: {:.4f}".format(valid_loss, valid_acc))
        if valid_acc > best_acc:
            best_acc = valid_acc
            food_classifier.save_checkpoint()
            img_encoder.save_checkpoint()

def test_model(img_encoder, food_classifier, test_loader):
    print("testing")
    food_classifier.load_checkpoint()
    img_encoder.load_checkpoint()
    food_classifier.eval()
    img_encoder.eval()
    with open(arglist.data_dir+'test.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['id', 'predicted'])
        for inputs, labels, paths in test_loader:
            inputs = inputs.to(food_classifier.device)
            food_classifier.optimizer.zero_grad()
            #features = extract_feat(img_encoder, inputs)
            #features = torch.reshape(features, (-1, arglist.input_size))
            outputs = food_classifier(inputs)
            pred = torch.softmax(outputs, dim=-1)
            _, predictions = torch.topk(pred, 3)
            predictions = predictions.cpu().numpy()
            pred = ""
            path = paths[0][paths[0].find("test_"):paths[0].find(".jpg")+4]
            for i in range(len(predictions[0])):
                pred += str(predictions[0][i])
                if i < len(predictions[0]):
                    pred += " "
            writer.writerow([path, pred])

if __name__ == '__main__':
    random.seed(0)
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ## about model
    num_classes = 1000
    input_size = 224

    ## data preparation
    data_reorganization.reorganize_data(data_path='../data', image_folder='train')
    data_reorganization.reorganize_data(data_path='../data', image_folder='val')
    data_reorganization.reorganize_data(data_path='../data', image_folder='test')
    train_loader, valid_loader, test_loader = data.load_data(data_dir=arglist.data_dir, input_size=input_size, batch_size=arglist.batch_size)
    ## model initialization
    img_encoder = MTFoodFeature(arglist.architecture, arglist.encoder_dir)
    img_encoder.eval()
    #img_encoder.cuda()
    food_classifier = MTFoodClassify(lr=arglist.lr, inpt_dims=arglist.input_size, fc1_dims=arglist.layer1_size, out_dims=arglist.num_classes, model_dir=arglist.model_dir)
    #food_classifier.cuda()

    ## loss function
    criterion = nn.CrossEntropyLoss()

    train_model(img_encoder, food_classifier, train_loader, valid_loader, criterion, num_epochs=arglist.num_epochs)

    test_model(img_encoder, food_classifier, test_loader)
