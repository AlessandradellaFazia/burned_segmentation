from typing import Any, Callable

import torch

from logic.losses.fn import DiceLoss, SoftBCEWithLogitsLoss
from logic.modules.base import BaseModule


class SingleTaskModule(BaseModule):
    def __init__(
        self,
        config: dict,
        tiler: Callable[..., Any] | None = None,
        predict_callback: Callable[..., Any] | None = None,
        loss: str = "bce",
        reprojected: bool = False,
        test_type: str = "standard",
    ):
        super().__init__(config, tiler, predict_callback, reprojected=reprojected)
        if loss == "bce":
            self.criterion_decode = SoftBCEWithLogitsLoss(
                ignore_index=255, pos_weight=torch.tensor(3.0)
            )
        else:
            self.criterion_decode = DiceLoss(
                mode="binary", from_logits=True, ignore_index=255
            )
        if test_type == "standard":
            self.predict_step = self.standard_predict_step
        else:
            self.predict_step = self.full_predict_step

    def training_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]  # batch,12,512,512
        y_del = batch["DEL"]  # batch, 512, 512

        # lc = batch["ESA_LC"]
        # x = torch.cat([x, lc.unsqueeze(1)], dim=1)
        decode_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss = loss_decode

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        for metric_name, metric in self.train_metrics.items():
            metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        # lc = batch["ESA_LC"]
        # x = torch.cat([x, lc.unsqueeze(1)], dim=1)
        decode_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss = loss_decode

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for metric_name, metric in self.val_metrics.items():
            metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):

        x = batch["S2L2A"]
        y_del = batch["DEL"]
        # lc = batch["ESA_LC"]
        # x = torch.cat([x, lc.unsqueeze(1)], dim=1)
        decode_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss = loss_decode

        self.log("test_loss", loss, on_epoch=True, logger=True)
        for metric_name, metric in self.test_metrics.items():
            metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, logger=True)
        return loss

    def full_predict_step(self, batch: Any, batch_idx: int):
        print("full predict step")

        full_image = batch["S2L2A"]  # [1, 12, h, w]
        y_del = batch["DEL"].squeeze(1)  # [1, 1, h, w]

        def callback(batch: Any):
            del_out = self.model(batch)  # [b, 1, h, w]
            return del_out.squeeze(1)  # [b, h, w]

        full_pred = self.tiler(full_image[0], callback=callback)
        full_pred = torch.sigmoid(full_pred)

        for metric_name, metric in self.test_metrics.items():
            metric(full_pred, y_del.float())
            self.log(metric_name, metric, on_epoch=True, logger=True)
        return

    def standard_predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        print("standard predict step")

        full_image = batch["S2L2A"]  # [1, 12, h, w]

        def callback(batch: Any):
            del_out = self.model(batch)  # [b, 1, h, w]
            return del_out.squeeze(1)  # [b, h, w]

        full_pred = self.tiler(
            full_image[0], callback=callback
        )  # full_image[0] 12, 1442, 1977
        batch["pred"] = torch.sigmoid(full_pred)
        return batch  # # matrice composta solo di 0 e 1

    def custom_predict_step(self, full_image: Any, dataloader_idx: int = 0) -> Any:

        def callback(batch: Any):
            del_out = self.model(batch)  # [b, 1, h, w]
            return del_out.squeeze(1)  # [b, h, w]

        full_pred = self.tiler(full_image[0], callback=callback)
        return torch.sigmoid(full_pred)

    def batched_predict_step(self, batch: Any, dataloader_idx: int = 0) -> Any:
        full_image = batch["S2L2A"]  # [b, 12, h, w]

        def callback(batch: Any):
            del_out = self.model(batch)  # [b, 1, h, w]
            return del_out.squeeze(1)  # [b, h, w]

        lista = []
        for image in full_image:
            pred = self.tiler(image, callback=callback)
            pred = torch.sigmoid(pred).unsqueeze(0)
            lista.append(pred)

        probs = torch.cat(lista, dim=0)
        return probs

    def on_predict_batch_end(
        self, outputs: Any | None, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.predict_callback(batch)
