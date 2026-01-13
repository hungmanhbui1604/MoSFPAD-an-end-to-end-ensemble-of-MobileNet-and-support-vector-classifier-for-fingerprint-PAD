import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


class FinPADDataset(Dataset):
    LABEL_MAP = {"Live": 0, "Spoof": 1}
    SUPPORTED_EXTENSIONS = (".png", ".bmp")

    def __init__(self, sensor_path, train=True):
        self.samples = []
        phase = "Train" if train else "Test"
        phase_path = os.path.join(sensor_path, phase)
        print(f"Loading {"train" if train else "test"} data of {sensor_path}")
        self._load_data(phase_path)

    def _load_data(self, phase_path):
        # Load Live samples
        live_path = os.path.join(phase_path, "Live")
        for img_file in tqdm(os.listdir(live_path), desc="Loading Live"):
            if img_file.endswith(self.SUPPORTED_EXTENSIONS):
                image_path = os.path.join(live_path, img_file)
                self.samples.append((image_path, self.LABEL_MAP["Live"]))

        # Load Spoof samples
        spoof_path = os.path.join(phase_path, "Spoof")
        for material in os.listdir(spoof_path):
            material_path = os.path.join(spoof_path, material)
            for img_file in tqdm(os.listdir(material_path), desc=f"Loading {material}"):
                if img_file.endswith(self.SUPPORTED_EXTENSIONS):
                    image_path = os.path.join(material_path, img_file)
                    self.samples.append((image_path, self.LABEL_MAP["Spoof"]))

    def __len__(self):
        return len(self.samples)

    @property
    def label_map(self):
        return self.LABEL_MAP

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return image, label


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def split_dataset(dataset: Dataset, val_split: float = 0.2, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)

    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    return random_split(dataset, [train_size, val_size], generator=generator)


def get_data_loaders(
    train_sensor_path: str,
    test_sensor_path: str,
    transform: dict,
    batch_size: int,
    num_workers: int = 0,
    val_split: float = 0.2,
    seed: int = 42,
):
    # create dataset
    if train_sensor_path == test_sensor_path:  # intra-sensor
        sensor_path = train_sensor_path
        print(f"Creating intra-sensor {sensor_path} dataset")
        train_dataset = FinPADDataset(sensor_path, train=True)
        test_dataset = FinPADDataset(sensor_path, train=False)
    else:  # cross-sensor
        print(f"Creating cross-sensor {train_sensor_path}-{test_sensor_path} dataset")
        train_dataset = FinPADDataset(train_sensor_path, train=True)
        test_dataset = FinPADDataset(test_sensor_path, train=False)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    # get label map
    label_map = train_dataset.label_map

    use_pin_memory = True if torch.cuda.is_available() else False
    # Train phase
    train_subset, val_subset = split_dataset(
        train_dataset, val_split=val_split, seed=seed
    )
    train_set = TransformedDataset(train_subset, transform["Train"])
    val_set = TransformedDataset(val_subset, transform["Test"])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    print(f"Number of train batches: {len(train_loader)}")
    print(f"Number of val batches: {len(val_loader)}")

    # Test phase
    test_set = TransformedDataset(test_dataset, transform["Test"])
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    print(f"Number of test batches: {len(test_loader)}")
    return train_loader, val_loader, test_loader, label_map
