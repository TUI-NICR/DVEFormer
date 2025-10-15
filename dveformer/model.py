# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict

from collections import ChainMap

import numpy as np
import torch

from nicr_mt_scene_analysis.model.backbone import get_backbone
from nicr_mt_scene_analysis.model.block import get_block_class
from nicr_mt_scene_analysis.model.context_module import get_context_module
from nicr_mt_scene_analysis.model.encoder import get_encoder
from nicr_mt_scene_analysis.model.encoder_decoder_fusion import get_encoder_decoder_fusion_class
from nicr_mt_scene_analysis.model.encoder_fusion import get_encoder_fusion_class
from nicr_mt_scene_analysis.model.upsampling import Upsampling
from nicr_mt_scene_analysis.model.initialization import he_initialization
from nicr_mt_scene_analysis.model.initialization import zero_residual_initialization

from .data import cast_to_dtype
from .data import DatasetConfig
from .data import DatasetConfigWithAuxiliary
from .decoder import get_decoders


class DVEFormer(torch.nn.Module):
    def __init__(
        self,
        args,
        dataset_config: DatasetConfig,
    ) -> None:
        super().__init__()

        # store args and dataset parameters
        self.args = args
        self.dataset_config = dataset_config

        # store auxiliary embeddings if available
        self.semantic_text_embeddings = None
        self.mean_visual_embeddings_per_semantic_class = None

        # get some dataset properties
        semantic_labels = dataset_config.semantic_label_list_without_void
        semantic_n_classes = len(semantic_labels)
        text_embeddings_per_semantic_class = None
        mean_visual_embedding_per_semantic_class = None

        scene_n_classes = len(dataset_config.scene_label_list_without_void)
        panoptic_semantic_classes_is_thing = semantic_labels.classes_is_thing
        panoptic_use_orientation = tuple(semantic_labels.classes_use_orientations)
        # only available with auxiliary data
        if isinstance(dataset_config, DatasetConfigWithAuxiliary):
            text_embeddings_per_semantic_class = dataset_config.semantic_text_embeddings
            if len(text_embeddings_per_semantic_class) <= 0:
                # If dataset was loaded without data, an empty list will be
                # returned. This makes problem with calls to partial_class
                # later on which is why we set to None.
                text_embeddings_per_semantic_class = None
            # determine a tensor of shape (n_classes, embedding_dim) which
            # contains the text embeddings per semantic class
            if text_embeddings_per_semantic_class is not None:
                # With void!
                assert len(text_embeddings_per_semantic_class) == semantic_n_classes + 1
                text_embeddings_per_semantic_class = torch.tensor(
                    np.array(text_embeddings_per_semantic_class), dtype=torch.float32
                )
                text_embeddings_per_semantic_class = text_embeddings_per_semantic_class[1:]
                text_embeddings_per_semantic_class = text_embeddings_per_semantic_class.to(args.device)
                self.semantic_text_embeddings = text_embeddings_per_semantic_class

            # compute mean visual embedding per semantic class and adjust
            # embedding with image_embedding to account for scene context.
            # shape is (n_classes, embedding_dim).
            # the resultig tensor can be used to determine the semantic
            # classes based on the predicted visual embedding
            mean_embedding_per_semantic_class = dataset_config.mean_embedding_per_semantic_class
            mean_image_embedding_per_semantic_class = dataset_config.mean_image_embedding_per_semantic_class
            if all((
                mean_embedding_per_semantic_class is not None,
                mean_image_embedding_per_semantic_class is not None
            )):
                mean_visual_embedding_per_semantic_class = torch.zeros_like(
                    text_embeddings_per_semantic_class
                ) if text_embeddings_per_semantic_class is not None else None

                if mean_visual_embedding_per_semantic_class is not None:
                    for semantic_idx, semantic_embedding in mean_embedding_per_semantic_class.items():
                        mean_image_embedding = torch.from_numpy(
                            mean_image_embedding_per_semantic_class[semantic_idx]
                        )
                        semantic_embedding = torch.from_numpy(semantic_embedding)

                        semantic_embedding_with_adjusted_scene_context = (
                            semantic_embedding - args.dense_visual_embedding_diff_factor * mean_image_embedding
                        )
                        semantic_embedding_with_adjusted_scene_context /= (
                            semantic_embedding_with_adjusted_scene_context.norm(dim=-1, keepdim=True)
                        )
                        # -1 because we do not have embedding for void class
                        mean_visual_embedding_per_semantic_class[semantic_idx-1] = \
                            semantic_embedding_with_adjusted_scene_context
                    self.mean_visual_embeddings_per_semantic_class = \
                        mean_visual_embedding_per_semantic_class

        # create encoder(s)
        if 'rgb' in args.input_modalities:
            backbone_rgb = get_backbone(
                name=args.rgb_encoder_backbone,
                resnet_block=get_block_class(
                    args.rgb_encoder_backbone_resnet_block,
                    dropout_p=args.dropout_p
                ),
                n_input_channels=3,
                normalization=args.encoder_normalization,
                activation=args.activation,
                pretrained=not args.no_pretrained_backbone,
                pretrained_filepath=args.rgb_encoder_backbone_pretrained_weights_filepath
            )
        else:
            backbone_rgb = None

        if 'depth' in args.input_modalities:
            backbone_depth = get_backbone(
                name=args.depth_encoder_backbone,
                resnet_block=get_block_class(
                    args.depth_encoder_backbone_resnet_block,
                    dropout_p=args.dropout_p
                ),
                n_input_channels=1,
                normalization=args.encoder_normalization,
                activation=args.activation,
                pretrained=not args.no_pretrained_backbone,
                pretrained_filepath=args.depth_encoder_backbone_pretrained_weights_filepath
            )
        else:
            backbone_depth = None

        if 'rgbd' in args.input_modalities:
            backbone_rgbd = get_backbone(
                name=args.rgbd_encoder_backbone,
                resnet_block=get_block_class(
                    args.rgbd_encoder_backbone_resnet_block,
                    dropout_p=args.dropout_p
                ),
                n_input_channels=3+1,
                normalization=args.encoder_normalization,
                activation=args.activation,
                pretrained=not args.no_pretrained_backbone,
                pretrained_filepath=args.rgbd_encoder_backbone_pretrained_weights_filepath
            )
        else:
            backbone_rgbd = None

        # fuse encoder(s) in a shared module
        self.encoder = get_encoder(
            backbone_rgb=backbone_rgb,
            backbone_depth=backbone_depth,
            backbone_rgbd=backbone_rgbd,
            fusion=args.encoder_fusion,
            normalization=args.encoder_normalization,
            activation=args.activation,
            skip_downsamplings=args.encoder_decoder_skip_downsamplings
        )
        enc_downsampling = self.encoder.downsampling
        enc_n_channels_out = self.encoder.n_channels_out
        enc_skips_n_channels = self.encoder.skips_n_channels

        # create context module
        self.context_module = get_context_module(
            name=args.context_module,
            n_channels_in=enc_n_channels_out,
            n_channels_out=enc_n_channels_out,
            input_size=(args.input_height // enc_downsampling,
                        args.input_width // enc_downsampling),
            # context module only makes sense with batch normalization
            normalization='bn',
            activation=args.activation,
            upsampling=args.upsampling_context_module
        )

        # create decoder(s)
        if args.instance_offset_encoding == 'tanh':
            instance_normalized_offset = True
            instance_tanh_for_offset = True
        elif args.instance_offset_encoding == 'relative':
            instance_normalized_offset = True
            instance_tanh_for_offset = False
        elif args.instance_offset_encoding == 'deeplab':
            instance_normalized_offset = False
            instance_tanh_for_offset = False
        else:
            raise NotImplementedError

        if args.instance_center_encoding == 'sigmoid':
            instance_sigmoid_for_center = True
        else:
            instance_sigmoid_for_center = False

        self.decoders = get_decoders(
            args,
            n_channels_in=enc_n_channels_out,
            downsampling_in=enc_downsampling,
            # semantic segmentation
            semantic_n_classes=semantic_n_classes,
            # instance segmentation
            instance_normalized_offset=instance_normalized_offset,
            instance_offset_distance_threshold=args.instance_offset_distance_threshold,
            instance_sigmoid_for_center=instance_sigmoid_for_center,
            instance_tanh_for_offset=instance_tanh_for_offset,
            # surface normal estimation
            normal_n_channels_out=3,
            # scene classification
            scene_n_channels_in=self.context_module.n_channels_reduction,
            scene_n_classes=scene_n_classes,
            # panoptic
            panoptic_semantic_classes_is_thing=panoptic_semantic_classes_is_thing,
            panoptic_has_orientation=panoptic_use_orientation,
            # embedding
            text_embeddings_per_class=text_embeddings_per_semantic_class,
            mean_visual_embeddings_per_class=mean_visual_embedding_per_semantic_class,
            # other shared args
            fusion_n_channels=enc_skips_n_channels[::-1],
        )

        # initialization
        debug_init = args.debug
        # apply he initialization to selected parts of the network
        for part in args.he_init:
            # whitelisted initialization
            cls = None
            if 'encoder-fusion' == part:
                cls = get_encoder_fusion_class(args.encoder_fusion)
            elif 'encoder-decoder-fusion' == part:
                cls = get_encoder_decoder_fusion_class(
                    args.encoder_decoder_fusion
                )

            if cls is not None:
                for n, m in self.named_modules():
                    if isinstance(m, cls):
                        he_initialization(m, name_hint=n, debug=debug_init)

            # (blacklisted) initialization
            if 'context-module' == part:
                he_initialization(self.context_module, debug=debug_init)
            elif 'decoder' == part:
                he_initialization(self.decoders,
                                  blacklist=(Upsampling,),
                                  debug=debug_init)

        # init last norm in residuals to zero to enforce identity on start
        if not args.no_zero_init_decoder_residuals:
            zero_residual_initialization(self.decoders, debug=debug_init)

        # Setup amp (automatic mixed precision)
        self.encoder_amp = torch.float32
        if args.encoder_amp == 'fp16':
            self.encoder_amp = torch.float16
        elif args.encoder_amp == 'bfp16':
            self.encoder_amp = torch.bfloat16
        else:
            assert args.encoder_amp == 'disabled'

        self.context_module_amp = torch.float32
        if args.context_module_amp == 'fp16':
            self.context_module_amp = torch.float16
        elif args.context_module_amp == 'bfp16':
            self.context_module_amp = torch.bfloat16
        else:
            assert args.context_module_amp == 'disabled'

        self.decoder_amp = torch.float32
        if args.decoder_amp == 'fp16':
            self.decoder_amp = torch.float16
        elif args.decoder_amp == 'bfp16':
            self.decoder_amp = torch.bfloat16
        else:
            assert args.decoder_amp == 'disabled'

    def compile(self):
        # Compile the model if requested. We only compile individual parts
        # instead of the whole model as if we compile the whole model, we
        # get some compiler issues.
        self.encoder = torch.compile(self.encoder)
        self.context_module = torch.compile(self.context_module)
        self.decoders = torch.compile(self.decoders)

    def forward(self, batch, do_postprocessing=False) -> Dict[str, Any]:
        # Setup encoder amp
        with torch.amp.autocast(
            device_type=self.args.device,
            enabled=self.encoder_amp != torch.float32 and self.training,
            dtype=self.encoder_amp if self.training else torch.float32
        ):
            # determine input
            enc_inputs = {}
            if 'rgbd' in self.args.input_modalities:
                rgb = batch['rgb']
                depth = batch['depth']
                rgbd = torch.cat([rgb, depth], dim=1)
                enc_inputs['rgbd'] = rgbd
            else:
                if 'rgb' in self.args.input_modalities:
                    enc_inputs['rgb'] = batch['rgb']
                if 'depth' in self.args.input_modalities:
                    enc_inputs['depth'] = batch['depth']
            # forward (fused) encoder(s)
            enc_outputs, enc_dec_skips = self.encoder(enc_inputs)

        # Cast the data types if amp types do not match
        if self.encoder_amp != self.context_module_amp:
            enc_outputs = cast_to_dtype(enc_outputs, self.context_module_amp)
            enc_dec_skips = \
                cast_to_dtype(enc_dec_skips, self.context_module_amp)

        # Setup context module amp
        with torch.amp.autocast(
            device_type=self.args.device,
            enabled=self.context_module_amp != torch.float32 and self.training,
            dtype=self.context_module_amp if self.training else torch.float32
        ):
            # forward context module
            if len(self.args.input_modalities) == 2:
                # design choice up to now, use output of rgb encoder as input for
                # context module
                con_input = enc_outputs['rgb']
            else:
                # use the output of the decoder with the same name as the input
                assert len(enc_inputs) == 1    # only one input modality
                con_input = enc_outputs[list(enc_inputs.keys())[0]]
            con_outputs, con_context_outputs = self.context_module(con_input)

        # Cast the data types if amp types do not match
        if self.context_module_amp != self.decoder_amp:
            con_outputs = cast_to_dtype(con_outputs, self.decoder_amp)
            con_context_outputs = \
                cast_to_dtype(con_context_outputs, self.decoder_amp)

        # Setup decoder amp
        with torch.amp.autocast(
            device_type=self.args.device,
            enabled=self.decoder_amp != torch.float32 and self.training,
            dtype=self.decoder_amp if self.training else torch.float32
        ):
            # forward decoder(s)
            outputs = []
            for decoder in self.decoders.values():
                outputs.append(
                    decoder(
                        (con_outputs, con_context_outputs), enc_dec_skips, batch,
                        do_postprocessing=do_postprocessing
                    )
                )

        # simplify output if postprocessing was applied
        if do_postprocessing:
            outputs = dict(ChainMap(*outputs))

        # TODO: Should we return outputs with mixed precision, or should we
        # cast everything to float32?

        return outputs
