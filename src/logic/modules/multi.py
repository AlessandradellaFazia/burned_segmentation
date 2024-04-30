from typing import Any, Callable
import torch
from loguru import logger
from torch import nn
from torchmetrics import F1Score, JaccardIndex, Precision, Recall

from logic.losses.fn import DiceLoss, SoftBCEWithLogitsLoss

from logic.modules.base import BaseModule


class MultiModule(BaseModule):
    def __init__(
        self,
        config: dict,
        tiler: Callable[..., Any] | None = None,
        predict_callback: Callable[..., Any] | None = None,
        reprojected=False,
        loss="bce",
        mask_lc=False,
    ):
        super().__init__(config, tiler, predict_callback, reprojected)
        if loss == "bce":
            self.criterion_decode = SoftBCEWithLogitsLoss(
                ignore_index=255, pos_weight=torch.tensor(3.0)
            )
        else:
            self.criterion_decode = DiceLoss(
                mode="binary", from_logits=True, ignore_index=255
            )
        self.criterion_aux = nn.CrossEntropyLoss(ignore_index=255)
        self.mask_lc = mask_lc
        self.aux_factor = config.decode_head.pop("aux_factor", 1.0)
        num_classes = config.decode_head.aux_classes
        logger.info(f"auxilary factor:{self.aux_factor}")
        logger.info(f"num classes:{num_classes}")
        self.train_metrics_aux = nn.ModuleDict(
            {
                "train_f1_aux": F1Score(
                    task="multiclass",
                    ignore_index=255,
                    num_classes=num_classes,
                    average="macro",
                ),
                "train_iou_aux": JaccardIndex(
                    task="multiclass",
                    ignore_index=255,
                    num_classes=num_classes,
                    average="macro",
                ),
            }
        )
        self.val_metrics_aux = nn.ModuleDict(
            {
                "val_f1_aux": F1Score(
                    task="multiclass",
                    ignore_index=255,
                    num_classes=num_classes,
                    average="macro",
                ),
                "val_iou_aux": JaccardIndex(
                    task="multiclass",
                    ignore_index=255,
                    num_classes=num_classes,
                    average="macro",
                ),
                "val_precision_aux": Precision(
                    task="multiclass",
                    ignore_index=255,
                    num_classes=num_classes,
                    average="macro",
                ),
                "val_recall_aux": Recall(
                    task="multiclass",
                    ignore_index=255,
                    num_classes=num_classes,
                    average="macro",
                ),
            }
        )
        self.test_metrics_aux = nn.ModuleDict(
            {
                "test_f1_aux": F1Score(
                    task="multiclass",
                    ignore_index=255,
                    num_classes=num_classes,
                    average="macro",
                ),
                "test_iou_aux": JaccardIndex(
                    task="multiclass",
                    ignore_index=255,
                    num_classes=num_classes,
                    average="macro",
                ),
                "test_precision_aux": Precision(
                    task="multiclass",
                    ignore_index=255,
                    num_classes=num_classes,
                    average="macro",
                ),
                "test_recall_aux": Recall(
                    task="multiclass",
                    ignore_index=255,
                    num_classes=num_classes,
                    average="macro",
                ),
            }
        )

    def training_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        y_lc = batch["ESA_LC"]
        if self.mask_lc:
            y_lc[y_del == 1] = 255
        decode_out, auxiliary_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss_auxiliary = self.criterion_aux(auxiliary_out, y_lc.long())
        loss = loss_decode + self.aux_factor * loss_auxiliary

        self.log(
            "train_loss_del",
            loss_decode,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss_aux",
            loss_auxiliary,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # compute delineation metrics
        for metric_name, metric in self.train_metrics.items():
            metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        # compute auxiliary metrics
        for metric_name, metric in self.train_metrics_aux.items():
            metric(auxiliary_out, y_lc.long())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        y_lc = batch["ESA_LC"]
        if self.mask_lc:
            y_lc[y_del == 1] = 255
        decode_out, auxiliary_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss_auxiliary = self.criterion_aux(auxiliary_out, y_lc.long())
        loss = loss_decode + self.aux_factor * loss_auxiliary

        self.log(
            "val_loss_del",
            loss_decode,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss_aux",
            loss_auxiliary,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # compute delineation metrics
        for metric_name, metric in self.val_metrics.items():
            metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        # compute auxiliary metrics
        for metric_name, metric in self.val_metrics_aux.items():
            metric(auxiliary_out, y_lc.long())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        x = batch["S2L2A"]
        y_del = batch["DEL"]
        y_lc = batch["ESA_LC"]
        if self.mask_lc:
            y_lc[y_del == 1] = 255
        decode_out, auxiliary_out = self.model(x)
        loss_decode = self.criterion_decode(decode_out.squeeze(1), y_del.float())
        loss_auxiliary = self.criterion_aux(auxiliary_out, y_lc.long())
        loss = loss_decode + self.aux_factor * loss_auxiliary

        self.log(
            "test_loss_del",
            loss_decode,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_loss_aux",
            loss_auxiliary,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # compute delineation metrics
        for metric_name, metric in self.test_metrics.items():
            metric(decode_out.squeeze(1), y_del.float())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        # compute auxiliary metrics
        for metric_name, metric in self.test_metrics_aux.items():
            metric(auxiliary_out, y_lc.long())
            self.log(metric_name, metric, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        full_image = batch["S2L2A"]

        def callback(batch: Any):
            del_out, _ = self.model(batch)  # [b, 1, h, w]
            return del_out.squeeze(1)  # [b, h, w]

        full_pred = self.tiler(full_image[0], callback=callback)
        batch["pred"] = torch.sigmoid(full_pred)
        return batch

    def on_predict_batch_end(
        self, outputs: Any | None, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.predict_callback(batch)