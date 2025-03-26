import torch
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image
import os
import pandas as pd
from collections import Counter


def main():
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    internal_idx = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5,
                    '14': 6, '15': 7, '16': 8, '17': 9, '18': 10, '19': 11,
                    '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '24': 17,
                    '25': 18, '26': 19, '27': 20, '28': 21, '29': 22, '3': 23,
                    '30': 24, '31': 25, '32': 26, '33': 27, '34': 28, '35': 29,
                    '38': 32, '39': 33, '4': 34, '40': 35, '41': 36, '42': 37,
                    '43': 38, '44': 39, '45': 40, '46': 41, '47': 42, '48': 43,
                    '49': 44, '5': 45, '50': 46, '51': 47, '52': 48, '53': 49,
                    '54': 50, '55': 51, '56': 52, '57': 53, '58': 54, '59': 55,
                    '6': 56, '60': 57, '61': 58, '62': 59, '63': 60, '64': 61,
                    '65': 62, '66': 63, '67': 64, '68': 65, '69': 66, '7': 67,
                    '70': 68, '71': 69, '72': 70, '73': 71, '74': 72, '75': 73,
                    '76': 74, '77': 75, '78': 76, '79': 77, '8': 78, '80': 79,
                    '81': 80, '82': 81, '83': 82, '84': 83, '85': 84, '86': 85,
                    '87': 86, '88': 87, '89': 88, '9': 89, '90': 90, '91': 91,
                    '92': 92, '93': 93, '94': 94, '95': 95, '96': 96, '97': 97,
                    '98': 98, '99': 99, '36': 30, '37': 31}

    models = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Resnext50 with more fc
    paths = [
        "/mnt/HDD3/home/owen/model/resnext101dropout/bestfreeze_0.9100.pth",
        "/mnt/HDD3/home/owen/model/resnext101dropout/bestfreeze_0.9067.pth",
        "/mnt/HDD3/home/owen/model/resnext101dropout/bestfreeze_0.9033.pth",
        # "/mnt/HDD3/home/owen/model/resnext101dropout/bestfreeze_0.9000.pth",
        # "/mnt/HDD3/home/owen/model/resnext101dropout/bestfreeze_200.pth",
        # "/mnt/HDD3/home/owen/model/resnext101dropout/bestfreeze_100.pth"
    ]
    for path in paths:
        model1 = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'resnext50_32x4d',
            weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2'
        )

        model1.fc = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 100)
        )

        model1.to(device)
        if torch.__version__ >= "2.0":
            model1 = torch.compile(model1)

        model1.load_state_dict(torch.load(path, map_location=device))
        models.append(model1)

    # Resnext101 with dropout
    paths = [
        "/mnt/HDD3/home/owen/model/resnext101dropout/bestfreeze_0.9133.pth",
        "/mnt/HDD3/home/owen/model/bestfreeze_0.9067.pth",
        "/mnt/HDD3/home/owen/model/bestfreeze_0.9100.pth",
    ]
    for path in paths:
        model2 = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'resnext101_32x8d',
            weights='ResNeXt101_32X8D_Weights.IMAGENET1K_V2'
        )

        model2.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 100)
        )
        model2.to(device)

        model2.load_state_dict(torch.load(path, map_location=device))
        models.append(model2)

    # Resnext50 with dropout
    paths = [
        "/mnt/HDD3/home/owen/model/resnext50dropout/bestfreeze_0.9133.pth",
        "/mnt/HDD3/home/owen/model/resnext50dropout/bestfreeze_0.9067.pth",
        "/mnt/HDD3/home/owen/model/resnext50dropout/bestfreeze_200.pth",
        "/mnt/HDD3/home/owen/model/resnext50dropout/bestfreeze_300.pth"
    ]
    for path in paths:
        model3 = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'resnext50_32x4d',
            weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2'
        )

        model3.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 100)
        )
        model3.to(device)

        model3.load_state_dict(torch.load(path, map_location=device))
        models.append(model3)

    # Resnext50
    paths = [
        "/mnt/HDD3/home/owen/model/resnext50dropout/bestfreeze_0.9100.pth",
        "/mnt/HDD3/home/owen/model/resnext50dropout/bestfreeze_0.9033.pth",
        # "/mnt/HDD3/home/owen/model/resnext50dropout/bestfreeze_0.9000.pth",
        # "/mnt/HDD3/home/owen/model/resnext50dropout/bestfreeze_0.8967.pth",
        # "/mnt/HDD3/home/owen/model/resnext50dropout/bestfreeze_0.8933.pth",
    ]
    for path in paths:
        model4 = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'resnext50_32x4d',
            weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2'
        )

        model4.fc = nn.Linear(2048, 100)
        if torch.__version__ >= "2.0":
            model4 = torch.compile(model4)

        model4.to(device)

        model4.load_state_dict(torch.load(path, map_location=device))
        models.append(model4)

    label = []
    allFilename = []
    test_root = '/mnt/HDD3/home/owen/data/test'

    for dirname, _, filenames in os.walk(test_root):
        for filename in sorted(filenames):
            filepath = os.path.join(dirname, filename)

            try:
                pic = Image.open(filepath).convert("RGB")
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                continue

            image = test_transform(pic).unsqueeze(0).to(device)

            preds = []
            for model in models:
                model.eval()
                with torch.no_grad():
                    outputs = model(image)
                    pred = outputs.argmax(dim=1).item()
                preds.append(pred)

            final_pred = Counter(preds).most_common(1)[0][0]

            label.append(
                [x for x, y in internal_idx.items() if y == final_pred][0]
            )
            allFilename.append(filename[:-4])

    df = pd.DataFrame({
        'image_name': allFilename,
        'pred_label': label
    })
    df.to_csv("prediction.csv", index=False)


if __name__ == "__main__":
    main()
