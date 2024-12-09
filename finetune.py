import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifiers import HAMBiomedCLIP
from datasets import HAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "./"
data_dir = os.path.join(root_dir, "data")

classifier = HAMBiomedCLIP(7, device=device)
classifier.train()

augmentations = T.Compose(
    [T.RandomVerticalFlip(), T.RandomHorizontalFlip(), T.RandomRotation(45)]
)
finetune_dataset = HAM(
    data_dir, finetune=True, transform=T.Compose([classifier.preprocess, augmentations])
)
val_dataset = HAM(data_dir, train=False, transform=classifier.preprocess)

finetune_dataloader = DataLoader(finetune_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-04, weight_decay=1e-05)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_accuracy = -torch.inf
for epoch in range(10):
    train_loss, train_correct = 0.0, 0
    for i, data in enumerate(tqdm(finetune_dataloader)):
        image, label = data

        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        cls_output = classifier(image)
        logits = cls_output["logits"]
        loss = F.cross_entropy(logits, label, reduction="sum")
        loss.backward()

        optimizer.step()

        prediction = torch.argmax(logits, dim=-1)
        correct = torch.sum(prediction == label)

        train_loss += loss.item()
        train_correct += correct.item()

    scheduler.step()

    val_loss, val_correct = 0.0, 0
    for i, data in enumerate(tqdm(val_dataloader)):
        image, label = data

        image = image.to(device)
        label = label.to(device)

        with torch.no_grad():
            cls_output = classifier(image)
            logits = cls_output["logits"]
            loss = F.cross_entropy(logits, label, reduction="sum")

            prediction = torch.argmax(logits, dim=-1)
            correct = torch.sum(prediction == label)

            val_loss += loss.item()
            val_correct += correct.item()

    val_accuracy = val_correct / len(val_dataset)
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        state_path = os.path.join(root_dir, "weights", "ham_biomedclip.pt")
        torch.save(classifier.state_dict(), state_path)

    print(f"Epoch {epoch + 1}:")
    print(f"\t Train loss: {train_loss / len(finetune_dataset):.4f}")
    print(f"\t Train accuracy: {train_correct / len(finetune_dataset):.2%}")
    print(f"\t Validation loss: {val_loss / len(val_dataset):.4f}")
    print(f"\t Validation accuracy: {val_correct / len(val_dataset):.2%}")
