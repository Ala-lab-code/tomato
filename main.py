from src.preprocess import split_dataset
from src.dataset import TomatoDataset

if __name__ == "__main__":


    train_dataset = TomatoDataset('data/processed/train', mode='train')
    print(f"Train dataset size: {len(train_dataset)}")
    img, label = train_dataset[0]
    print(f"Sample img shape: {img.shape}, label: {label}")
