from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from logic.datasets import EMSCropDataset, EMSImageDataset
from logic.samplers.samplers import SequentialTiledSampler, RandomTiledSampler

# from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler da installare


class EMSDataModule(LightningDataModule):
    transform_targets = {
        "S2L2A": "image",
        "DEL": "mask",
        "CM": "mask",
        "GRA": "mask",
        "ESA_LC": "mask",
    }

    def __init__(
        self,
        root: Path,
        patch_size: int = 512,
        modalities: list[str] = ["S2L2A", "DEL", "ESA_LC", "CM"],
        batch_size_train: int = 8,
        batch_size_eval: int = 16,
        num_workers: int = 2,
        derivative_idx=False,
        full_test_type: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.modalities = modalities
        self.patch_size = patch_size
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.derivative_idx = derivative_idx
        self.full_test_type = full_test_type
        self.train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(rotate_limit=360, value=0, mask_value=255, p=0.5),
                A.RandomBrightnessContrast(
                    p=0.5, brightness_limit=0.02, contrast_limit=0.02
                ),
                ToTensorV2(),
            ],
            additional_targets=self.transform_targets,
        )
        self.eval_transform = A.Compose(
            [ToTensorV2()],
            additional_targets=self.transform_targets,
        )

    def setup(self, stage=None):
        if stage == "fit":
            self.train_set = EMSCropDataset(
                root=self.root,
                subset="train",
                modalities=self.modalities,
                transform=self.train_transform,
                derivative_idx=self.derivative_idx,
            )
            self.val_set = EMSCropDataset(
                root=self.root,
                subset="val",
                modalities=self.modalities,
                transform=self.eval_transform,
                derivative_idx=self.derivative_idx,
            )
        elif stage == "test":
            if self.full_test_type:
                self.test_set = EMSImageDataset(
                    root=self.root,
                    subset="test",
                    modalities=self.modalities,
                    transform=self.eval_transform,
                    derivative_idx=self.derivative_idx,
                )
            else:
                self.test_set = EMSCropDataset(
                    root=self.root,
                    subset="test",
                    modalities=self.modalities,
                    transform=self.eval_transform,
                    derivative_idx=self.derivative_idx,
                )
        elif stage == "predict":
            self.pred_set = EMSImageDataset(
                root=self.root,
                subset="test",
                modalities=self.modalities,
                transform=self.eval_transform,
                derivative_idx=self.derivative_idx,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            sampler=RandomTiledSampler(self.train_set, tile_size=self.patch_size),
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            sampler=SequentialTiledSampler(
                self.val_set,
                tile_size=self.patch_size,
            ),
            batch_size=self.batch_size_eval,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.full_test_type:
            return DataLoader(
                self.test_set,
                shuffle=False,
                batch_size=1,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        return DataLoader(
            self.test_set,
            sampler=SequentialTiledSampler(
                self.test_set,
                tile_size=self.patch_size,
            ),
            batch_size=self.batch_size_eval,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_set,
            shuffle=False,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def batch_predict_dataloader(self, batch_size=4):
        return DataLoader(
            self.test_set,
            sampler=SequentialTiledSampler(
                self.test_set,
                tile_size=self.patch_size,
            ),
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
