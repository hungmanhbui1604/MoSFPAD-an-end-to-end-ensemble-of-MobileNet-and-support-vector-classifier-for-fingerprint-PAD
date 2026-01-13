from torchvision import transforms


def get_transforms(transform_type: str):
    if transform_type == "mosfpad":
        return {
            "train": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=0, shear=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        }
    else:
        return None 
