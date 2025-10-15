# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Tuple, Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F

from nicr_mt_scene_analysis.data.preprocessing.multiscale_supervision import get_downscale
from nicr_mt_scene_analysis.data.preprocessing.resize import get_fullres
from nicr_mt_scene_analysis.data.preprocessing.resize import get_fullres_key
from nicr_mt_scene_analysis.data.preprocessing.resize import get_valid_region_slices_and_fullres_shape
from nicr_mt_scene_analysis.loss import CrossEntropyLossSemantic
from nicr_mt_scene_analysis.metric import MeanIntersectionOverUnion
from nicr_mt_scene_analysis.model.postprocessing.dense_base import DensePostprocessingBase
from nicr_mt_scene_analysis.task_helper.base import append_detached_losses_to_logs
from nicr_mt_scene_analysis.task_helper.base import append_profile_to_logs
from nicr_mt_scene_analysis.task_helper.base import TaskHelperBase
from nicr_mt_scene_analysis.types import BatchType
from nicr_mt_scene_analysis.visualization import visualize_heatmap_pil
from nicr_mt_scene_analysis.visualization import visualize_semantic_pil

from .model import DVEFormer


class DVEFormerLinearProbing(DVEFormer):

    def __init__(
        self,
        args,
        dataset_config,
    ) -> None:
        super().__init__(args, dataset_config=dataset_config)

        # freeze base model parameters
        for param in super().parameters():
            param.requires_grad_(False)

        # required for batch norm so we don't alter the stats
        for module in self.modules():
            if hasattr(module, 'track_running_stats'):
                module.track_running_stats = False

        in_dim = self.decoders.dense_visual_embedding_decoder._embedding_dim
        out_dim = len(dataset_config.semantic_label_list_without_void)

        # We don't use bias so the weights are drop in replacement for
        # text-based and visual mean-based segmentation.
        self.linear_probing_head = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            bias=False
        )
        self._linear_postprocessing = LinearProbingPostprocessing()

        # Initialize the linear probing head
        init_mode = args.linear_probing_weights_init
        if init_mode == 'random':
            pass
        elif init_mode == 'text':
            device = self.linear_probing_head.weight.device
            text_embeddings = self.semantic_text_embeddings
            if text_embeddings is None:
                raise ValueError(
                    "Linear probing text initialization requested but no "
                    "semantic text embeddings available."
                )
            weight = F.normalize(text_embeddings.to(device), dim=1)[:, :, None, None]
            if weight.shape != self.linear_probing_head.weight.shape:
                raise ValueError(
                    "Prepared linear probing weights do not match head shape. "
                    "Falling back to random initialization."
                )
            self.linear_probing_head.weight.data.copy_(weight)
        elif init_mode == 'mean':
            device = self.linear_probing_head.weight.device
            mean_embeddings = self.mean_visual_embeddings_per_semantic_class
            if mean_embeddings is None:
                raise ValueError(
                    "Linear probing mean initialization requested but mean "
                    "visual embeddings are unavailable."
                )
            weight = mean_embeddings.to(device)[:, :, None, None]
            if weight.shape != self.linear_probing_head.weight.shape:
                raise ValueError(
                    "Prepared linear probing weights do not match head shape. "
                    "Falling back to random initialization."
                )
            self.linear_probing_head.weight.data.copy_(weight)
        else:
            raise ValueError(
                f"Unsupported linear probing init mode: '{init_mode}'."
            )

    def forward(self, batch, do_postprocessing: bool = True):
        # forward the original DVEFormer model
        with torch.no_grad():
            r_dict = super().forward(
                batch, do_postprocessing=do_postprocessing
            )

        # get the dense visual embeddings
        embeddings = r_dict['dense_visual_embedding_output']

        # detach to avoid gradients flowing into the base model
        embeddings = embeddings.detach()

        # ensure float32 for the linear probing head
        if embeddings.dtype != torch.float32:
            embeddings = embeddings.float()

        lp_embeddings = F.normalize(embeddings, dim=1)

        # apply linear probing
        output = self.linear_probing_head(lp_embeddings)

        if do_postprocessing:
            postprocessed = self._linear_postprocessing.postprocess(
                data=(output, tuple()),
                batch=batch,
                is_training=self.training
            )
            r_dict.update(postprocessed)
        else:
            r_dict['linear_probing_semantic'] = output
            r_dict['linear_probing_side_outputs'] = tuple()

        return r_dict


class LinearProbingPostprocessing(DensePostprocessingBase):

    def _postprocess_training(self, data, batch):
        output, side_outputs = data
        return {
            'linear_probing_semantic': output,
            'linear_probing_side_outputs': side_outputs
        }

    def _postprocess_inference(self, data, batch):
        output, side_outputs = data
        r_dict = {
            'linear_probing_semantic': output,
            'linear_probing_side_outputs': side_outputs
        }

        pred = F.softmax(output, dim=1)
        score, idx = torch.max(pred, dim=1)
        r_dict.update({
            'linear_probing_softmax_scores': pred,
            'linear_probing_semantic_score': score,
            'linear_probing_semantic_idx': idx,
        })

        crop_slices, resize_shape = \
            get_valid_region_slices_and_fullres_shape(batch, 'semantic')
        output_fullres = self._crop_to_valid_region_and_resize_prediction(
            output, valid_region_slices=crop_slices,
            shape=resize_shape, mode='bilinear'
        )
        pred_fullres = F.softmax(output_fullres, dim=1)
        score_fullres, idx_fullres = torch.max(pred_fullres, dim=1)
        r_dict.update({
            get_fullres_key('linear_probing_semantic'): output_fullres,
            get_fullres_key('linear_probing_softmax_scores'): pred_fullres,
            get_fullres_key('linear_probing_semantic_score'): score_fullres,
            get_fullres_key('linear_probing_semantic_idx'): idx_fullres
        })
        return r_dict


class LinearProbingTaskHelper(TaskHelperBase):

    def __init__(
        self,
        head: nn.Module,
        n_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        examples_cmap: Union[Sequence[Tuple[int, int, int]], torch.Tensor] = None
    ) -> None:
        super().__init__()
        self._head = head
        self._n_classes = n_classes
        self._class_weights = class_weights
        self._label_smoothing = label_smoothing
        self._loss = None
        self._metric_iou = None

        # during validation, we store some examples for visualization purposes
        self._examples = {}
        self._examples_cmap = examples_cmap

    def initialize(self, device: torch.device):
        self._head = self._head.to(device)
        self._head.train()
        weights = self._class_weights
        if weights is not None:
            weights = weights.to(device)
        self._loss = CrossEntropyLossSemantic(
            weights=weights,
            label_smoothing=self._label_smoothing
        )
        self._metric_iou = MeanIntersectionOverUnion(n_classes=self._n_classes)
        self._metric_iou.reset()

    def _get_spatial_target_for_prediction(
        self,
        batch: BatchType,
        batch_key: str,
        prediction: torch.Tensor
    ) -> torch.Tensor:
        target_fullres = batch[batch_key]
        h_target, w_target = target_fullres.shape[-2:]
        h_pred, w_pred = prediction.shape[-2:]

        if h_pred == h_target and w_pred == w_target:
            return target_fullres

        assert (h_target % h_pred) == 0 and (w_target % w_pred) == 0, (
            "Prediction and target resolutions are incompatible: "
            f"{(h_pred, w_pred)} vs {(h_target, w_target)}"
        )

        downscale_h = h_target // h_pred
        downscale_w = w_target // w_pred

        assert downscale_h == downscale_w, (
            "Non-uniform scaling between height and width is not supported: "
            f"{downscale_h} vs {downscale_w}"
        )

        downscale_sample = get_downscale(batch, downscale_h)
        assert downscale_sample is not None and batch_key in downscale_sample, (
            f"Required downscale '{downscale_h}' for key '{batch_key}' "
            "is missing in batch. Ensure multiscale preprocessing is enabled."
        )
        return downscale_sample[batch_key]

    def _compute_losses(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Dict[str, torch.Tensor]:

        preds, keys, _ = self.collect_predictions_for_loss(
            predictions_post=predictions_post,
            predictions_post_key='linear_probing_semantic',
            side_outputs_key=None  # No multiscale outputs
        )

        # collect the target in the correct resolution. the function will
        # always try to find the resolutions matching the output resolution
        # of the model.
        targets = [
            self._get_spatial_target_for_prediction(
                batch=batch,
                batch_key='semantic',
                prediction=pred
            )
            for pred in preds
        ]

        loss_outputs = self._loss(
            input_tensors=preds, target_tensors=targets
        )

        # create loss dict
        loss_dict = {
            f'linear_probing_loss_{key}': loss/n
            for key, (loss, n) in zip(keys, loss_outputs)
        }

        # compute total loss (accumulate losses of all side outputs)
        total_loss = self.accumulate_losses(
            losses=[loss for loss, _ in loss_outputs],
            n_elements=[n for _, n in loss_outputs]
        )

        loss_dict[self.mark_as_total('linear_probing')] = total_loss

        return loss_dict

    def parameters(self, recurse: bool = True):
        return self._head.parameters()

    @append_profile_to_logs('linear_probing_step_time')
    @append_detached_losses_to_logs()
    def training_step(
        self,
        batch,
        batch_idx: int,
        predictions_post
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        embeddings = predictions_post['dense_visual_embedding_output']
        # avoid gradients flowing into the base model
        embeddings = embeddings.detach()

        # ensure float32 for the linear probing head
        if embeddings.dtype != torch.float32:
            embeddings = embeddings.float()
        embeddings = F.normalize(embeddings, dim=1)
        logits = self._head(embeddings)
        loss_dict = self._compute_losses(
            batch=batch,
            batch_idx=batch_idx,
            predictions_post={'linear_probing_semantic': logits}
        )
        return loss_dict, {}

    @append_profile_to_logs('linear_probing_step_time')
    @append_detached_losses_to_logs()
    def validation_step(
        self,
        batch,
        batch_idx: int,
        predictions_post
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # compute loss
        loss_dict = self._compute_losses(
            batch=batch,
            batch_idx=batch_idx,
            predictions_post=predictions_post
        )

        # update metrics
        target_fullres = get_fullres(batch, 'semantic')
        if target_fullres is None:
            target = self._get_spatial_target_for_prediction(
                batch=batch,
                batch_key='semantic',
                prediction=predictions_post['linear_probing_semantic_idx']
            )
        else:
            target = target_fullres
        mask = target != 0    # mask of non-void pixels
        preds = predictions_post[
            get_fullres_key('linear_probing_semantic_idx')
        ][mask]
        target = target[mask] - 1    # first apply mask -> -1 is safe
        self._metric_iou.update(preds=preds.cpu(), target=target.cpu())

        # store example for visualization (not fullres!)
        if batch_idx == 0:
            # class
            ex = predictions_post['linear_probing_semantic_idx'][0]
            key = f'linear_probing_example_{batch_idx}_0'
            self._examples[key] = visualize_semantic_pil(
                semantic_img=ex.cpu().numpy(),
                colors=self._examples_cmap
            )

            # score
            ex = predictions_post['linear_probing_semantic_score'][0]
            key = f'linear_probing_example_score_{batch_idx}_0'
            self._examples[key] = visualize_heatmap_pil(
                heatmap_img=ex.cpu().numpy(),
                min_=0, max_=1
            )

        return loss_dict, {}

    @append_profile_to_logs('linear_probing_epoch_end_time')
    def validation_epoch_end(self):
        miou, ious = self._metric_iou.compute(return_ious=True)
        logs = {'linear_probing_miou': miou}
        artifacts = {'linear_probing_cm': self._metric_iou.confmat.clone(),
                     'linear_probing_ious_per_class': ious.clone()}

        # reset metric (it is not done automatically)
        self._metric_iou.reset()
        assert self._metric_iou.confmat.sum() == 0

        return artifacts, self._examples, logs


def get_linear_probing_task_helper(
    args, dataset, model: DVEFormerLinearProbing
) -> LinearProbingTaskHelper:
    assert isinstance(model, DVEFormerLinearProbing)

    class_weights: Optional[torch.Tensor] = None
    if args.semantic_class_weighting != 'none':
        weights_np = dataset.semantic_compute_class_weights(
            weight_mode=args.semantic_class_weighting,
            c=args.semantic_class_weighting_logarithmic_c,
            n_threads=4,
            debug=False
        )
        class_weights = torch.tensor(weights_np, dtype=torch.float)
    return LinearProbingTaskHelper(
        head=model.linear_probing_head,
        n_classes=dataset.semantic_n_classes_without_void,
        class_weights=class_weights,
        label_smoothing=args.semantic_loss_label_smoothing,
        examples_cmap=dataset.semantic_class_colors_without_void
    )
