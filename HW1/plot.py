import pickle
import matplotlib.pyplot as plt
import os


def main():
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    with open(path + r'\resnext50_32x4d-1-30.pickle', 'rb') as handle:
        acc_data_50_1 = pickle.load(handle)
    with open(path + r'\resnext50_32x4d-2-30.pickle', 'rb') as handle:
        acc_data_50_2 = pickle.load(handle)
    with open(path + r'\resnext101_32x8d-1-30.pickle', 'rb') as handle:
        acc_data_101_1 = pickle.load(handle)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 31), acc_data_50_1['train_acc'],
             label='Train Acc', marker='o', color='red')
    plt.plot(range(1, 31), acc_data_50_1['val_acc'],
             label='Val Acc', marker='o', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy (resnext50_32x4d-1)')
    plt.legend()
    plt.savefig(path + r'\plot_acc_50_1.jpg')

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 31), acc_data_50_1['train_loss'],
             label='Train Loss', marker='o', color='red')
    plt.plot(range(1, 31), acc_data_50_1['val_loss'],
             label='Val Loss', marker='o', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss (resnext50_32x4d-1)')
    plt.legend()
    plt.savefig(path + r'\plot_loss_50_1.jpg')

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 31), acc_data_50_2['train_acc'],
             label='Train Acc', marker='o', color='red')
    plt.plot(range(1, 31), acc_data_50_2['val_acc'],
             label='Val Acc', marker='o', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy (resnext50_32x4d-2)')
    plt.legend()
    plt.savefig(path + r'\plot_acc_50_2.jpg')

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 31), acc_data_50_2['train_loss'],
             label='Train Loss', marker='o', color='red')
    plt.plot(range(1, 31), acc_data_50_2['val_loss'],
             label='Val Loss', marker='o', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss (resnext50_32x4d-2)')
    plt.legend()
    plt.savefig(path + r'\plot_loss_50_2.jpg')

    plt.figure(figsize=(8, 5))
    plt.plot(range(15, 45), acc_data_101_1['train_acc'],
             label='Train Acc', marker='o', color='red')
    plt.plot(range(15, 45), acc_data_101_1['val_acc'],
             label='Val Acc', marker='o', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy (resnext101_32x8d-1-30)')
    plt.legend()
    plt.savefig(path + r'\plot_acc_101_1.jpg')

    plt.figure(figsize=(8, 5))
    plt.plot(range(15, 45), acc_data_101_1['train_loss'],
             label='Train Loss', marker='o', color='red')
    plt.plot(range(15, 45), acc_data_101_1['val_loss'],
             label='Val Loss', marker='o', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss (resnext101_32x8d-1-30)')
    plt.legend()
    plt.savefig(path + r'\plot_loss_101_1.jpg')

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 31), acc_data_50_1['val_acc'],
             label='ResNeXt50_32x4d(Dropout Fc)', marker='o', color='red')
    plt.plot(range(1, 31), acc_data_50_2['val_acc'],
             label='ResNeXt50_32x4d(BatchNorm + Dropout Fc)',
             marker='o', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Dropout Fc vs BatchNorm + Dropout Fc')
    plt.legend()
    plt.savefig(path + r'\plot_acc_comp.jpg')

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 31), acc_data_50_1['val_loss'],
             label='ResNeXt50_32x4d(Dropout Fc)', marker='o', color='red')
    plt.plot(range(1, 31), acc_data_50_2['val_loss'],
             label='ResNeXt50_32x4d(BatchNorm + Dropout Fc)',
             marker='o', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Dropout Fc vs BatchNorm + Dropout Fc')
    plt.legend()
    plt.savefig(path + r'\plot_loss_comp.jpg')


if __name__ == "__main__":
    main()
    # load()
