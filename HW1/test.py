
from collections import Counter
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision.datasets as datasets
import tqdm
import numpy as np


def main(load_checkpoint=False, model_name='resnext50_32x4d', fc_type=0,
         unfreeze_num=4, num_epochs=100, train_batch=100, compile=False,
         checkpoint_dir='', save_dir=''):

    # Load Training and Validation data
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.RandomResizedCrop(224, scale=(0.6, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(20),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        v2.RandomErasing(p=0.3),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = '/mnt/HDD3/home/owen/data/train'
    val_dir = '/mnt/HDD3/home/owen/data/val'
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transform
    )

    print("Class to index mapping (folder -> class index):")
    print(train_dataset.class_to_idx)

    print("\nSample file paths and labels:")
    for i in range(10):  # Check first 10 samples
        path, label = train_dataset.samples[i]
        print(f"Path: {path}, Label: {label}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch,
        shuffle=True,
        num_workers=12
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=20,
        shuffle=False,
        num_workers=4
    )

    # Calculate the weight of each class
    class_counts = Counter(train_dataset.targets)
    weight = np.zeros(100)
    for cls_idx, count in class_counts.items():
        weight[cls_idx] = 1 / count

    # Build the model
    model_weight = {
        'resnext50_32x4d': 'ResNeXt50_32X4D_Weights.IMAGENET1K_V2',
        'resnext101_32x8d': 'ResNeXt101_32X8D_Weights.IMAGENET1K_V2'
    }
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name,
                           weights=model_weight[model_name])

    if fc_type == 1:
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 100)
        )
    elif fc_type == 2:
        model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 100)
        )
    else:
        model.fc = nn.Linear(2048, 100)

    found_name = False
    for name, params in model.named_parameters():
        if name == f"layer{unfreeze_num}.0.conv1.weight":
            found_name = True
        params.requires_grad = found_name

    if compile and torch.__version__ >= "2.0":
        model = torch.compile(model)

    if load_checkpoint:
        model.load_state_dict(torch.load(checkpoint_dir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_weights = torch.tensor(weight, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.00001,
        weight_decay=1e-4
    )

    class_weights = torch.tensor(np.float64(weight)).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    # Training loop
    maxAcc = -999
    AccHistory = {'train_acc': [], 'val_acc': [],
                  'train_loss': [], 'val_loss': []}
    for epoch in range(0, num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        for i, (images, labels) in tqdm(enumerate(train_loader),
                                        total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        scheduler.step()

        accuracy = total_correct / len(train_dataset)
        train_loss = total_loss / len(train_loader)

        epoch_p = f"[Epoch {epoch+1}]"
        t_p = f"Train Loss: {train_loss:.4f}, Train Accuracy: {accuracy:.4f}"
        print(epoch_p + t_p)

        # Validation loop
        model.eval()
        val_correct = 0
        total_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        val_accuracy = val_correct / len(val_dataset)
        val_loss = total_loss / len(val_loader)
        val_p1 = f"Validation Loss: {val_loss:.4f},"
        val_p2 = f" Validation Accuracy: {val_accuracy:.4f}"
        print(val_p1 + val_p2)

        AccHistory['train_acc'].append(accuracy)
        AccHistory['val_acc'].append(val_accuracy)
        AccHistory['train_loss'].append(train_loss)
        AccHistory['val_loss'].append(val_loss)

        if epoch % 100 == 0:
            torch.save(model.state_dict(),
                       f"{save_dir}/bestfreeze_{epoch}.pth")
            with open(f'{model_name} - {epoch}.pickle', 'wb') as handle:
                pickle.dump(AccHistory,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        if val_accuracy > maxAcc:
            maxAcc = val_accuracy
            torch.save(model.state_dict(),
                       f"{save_dir}/bestfreeze_{maxAcc:.4f}.pth")

        torch.save(model.state_dict(), "test.pth")

    with open(f'{model_name}-{fc_type}-{num_epochs}.pickle', 'wb') as handle:
        pickle.dump(AccHistory, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    check_dir = "./model/resnext50dropout/bestfreeze_0.8733.pth"
    save_dir = "./model/resnext50dropout"
    main(
        load_checkpoint=False,
        model_name="resnext50_32x4d",
        fc_type=2,
        unfreeze_num=2,
        num_epochs=30,
        train_batch=100,
        compile=False,
        checkpoint_dir=check_dir,
        save_dir=save_dir
    )

    # load()
