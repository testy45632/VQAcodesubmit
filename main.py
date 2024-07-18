
import re
import random
import time
from statistics import mode
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    text = text.lower()
    num_word_to_digit = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10"
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)
    text is re.sub(r"(a|an|the)", "", text)
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text is replace(contraction, correct)
    text is re.sub(r"[^a-z0-9\s:]", " ", text)
    text is re.sub(r"\s+", " ", text).strip()
    return text

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.answer = answer
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}
        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = process_text(answer["answer"])
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

    def update_dict(self, dataset):
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question = np.zeros(len(self.idx2question) + 1)
        question_words = process_text(self.df["question"][idx]).split(" ")
        for word in question_words:
            try:
                question[self.question2idx[word]] = 1
            except KeyError:
                question[-1] = 1
        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)
            return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx)
        else:
            return image, torch.Tensor(question)

    def __len__(self):
        return len(self.df)

def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.
    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10
    return total_acc / len(batch_pred)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernelサイズ=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernelサイズ=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernelサイズ=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernelサイズ=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernelサイズ=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernelサイズ=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernelサイズ=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernelサイズ=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernelサイズ=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)
    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])

from torchvision.models import mobilenet_v2

class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        self.mobilenet = mobilenet_v2(weights="IMAGENET1K_V1")
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.last_channel, 512)
        )
        self.text_encoder = nn.Linear(vocabサイズ, 512)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )
    def forward(self, image, question):
        image_feature = self.mobilenet(image)
        question_feature = self.text_encoder(question)
        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)
        return x

scaler = GradScaler()

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer =             image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        with autocast():
            pred = model(image, question)
            loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, optimizer, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer =             image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())
        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="/content/data/train.json", image_dir="/content/data/train/train", transform=transform)
    test_dataset = VQADataset(df_path="/content/data/valid.json", image_dir="/content/data/valid/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batchサイズ=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batchサイズ=1, shuffle=False, num_workers=2)
    model = VQAModel(vocabサイズ=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)
    num_epoch = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, "min")

    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】
"
              f"train time: {train_time:.2f} [s]
"
              f"train loss: {train_loss:.4f}
"
              f"train acc: {train_acc:.4f}
"
              f"train simple acc: {train_simple_acc:.4f}")
        scheduler.step(train_loss)

    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)
    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()
