import torch
import numpy as np
from pathlib import Path
import glob
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import os
import spacy
from transformers import BertTokenizer
random.seed(10)

class CustomDataset(Dataset):
    def __init__(self, directory='./custom_dataset/', train=True, ratio=0.7, transform=None):
        self.labels = []
        self.route = []
        self.transform = transform
        self.directory_path = Path(directory)
        self.classes = [str(file) for file in self.directory_path.iterdir() if file.is_dir()]
        self.mode = 'image'
        nlp = spacy.load('en_core_web_sm')

        for n, c in enumerate(self.classes):
            filenames = list(glob.glob(c+"/*"))
            _, file_extension = os.path.splitext(filenames[0])
            if file_extension in ['.png', '.jpg', 'jpeg', '.PNG', '.JPG', '.JPEG']:
                filenames = list(glob.glob(c+"/*"))
                self.route += filenames
                self.labels += [n] * len(filenames)
            elif file_extension == '.pdf':
                self.mode = 'text'
                for j, file_name in enumerate(filenames):
                    if j % 100 == 0:
                        print('loading...')
                    reader = PdfReader(file_name)
                    for page in reader.pages:
                        file_doc = nlp(page.extract_text())
                        sentences = list(file_doc.sents)
                        list_of_sentences = [i.text for i in sentences]
                        self.route += list_of_sentences
                        self.labels += [n] * len(sentences)

            elif file_extension == '.txt':
                self.mode = 'text'
                for j, file_name in enumerate(filenames):
                    if j % 100 == 0:
                        print('loading...')
                    file_text = open(file_name).read()
                    file_doc = nlp(file_text)
                    sentences = list(file_doc.sents)
                    list_of_sentences = [i.text for i in sentences]
                    self.route += list_of_sentences
                    self.labels += [n] * len(sentences)

        last = int(len(self.labels) * ratio)
        pack = list(zip(self.labels, self.route))
        random.shuffle(pack)
        self.labels, self.route = zip(*pack)
        if train:
            self.labels = self.labels[:last]
            self.route = self.route[:last]
            print('Total train examples:',last)
        else:
            self.labels = self.labels[last:]
            self.route = self.route[last:]
            print('Total test examples:',len(self.labels))
        if self.mode == 'text':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.route = [tokenizer(text, 
                            padding='max_length', max_length=512, truncation=True,
                            return_tensors="pt") for text in self.route]


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        filename = self.route[index]
        if self.mode == 'text':
            x = filename
        else:
            x = Image.open(filename).convert('RGB')
        label = self.labels[index]
        label = np.array(label)

        if self.transform:
            x = self.transform(x)

        example = [x, label]
        return example

    def get_class_name(self):
        return [os.path.basename(os.path.normpath(classname)) for classname in self.classes]

class UnlabeledDataset(Dataset):
    def __init__(self, directory='./unlabeled_dataset/', train=True, ratio=0.7, transform=None, augment=None):
        self.route = []
        self.transform = transform
        self.augment = augment
        self.directory_path = Path(directory)
        self.filenames = list(glob.glob(self.directory_path))
        self.route += self.filenames
        last = int(len(self.route) * ratio)
        random.shuffle(self.route)

        if train:
            self.route = self.route[:last]
        else:
            self.route = self.route[last:]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        filename = self.route[index]
        image = Image.open(filename).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.augment:
            aug = self.augment(image)
            return [image, arg]

        return [image, None]