# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from glob import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from nicr_mt_scene_analysis.data import move_batch_to_device
from nicr_mt_scene_analysis.data import mt_collate

from dveformer.args import ArgParserDVEFormer
from dveformer.data import get_datahelper
from dveformer.linear_probing import DVEFormerLinearProbing
from dveformer.model import DVEFormer
from dveformer.preprocessing import get_preprocessor
from dveformer.visualization import visualize_predictions
from dveformer.weights import load_weights


def _get_args():
    parser = ArgParserDVEFormer()

    # add additional arguments
    group = parser.add_argument_group('Inference')
    group.add_argument(    # useful for appm context module
        '--inference-input-height',
        type=int,
        default=480,
        dest='validation_input_height',    # used in test phase
        help="Network input height for predicting on inference data."
    )
    group.add_argument(    # useful for appm context module
        '--inference-input-width',
        type=int,
        default=640,
        dest='validation_input_width',    # used in test phase
        help="Network input width for predicting on inference data."
    )
    group.add_argument(
        '--depth-max',
        type=float,
        default=None,
        help="Additional max depth values. Values above are set to zero as "
             "they are most likely not valid. Note, this clipping is applied "
             "before scaling the depth values."
    )
    group.add_argument(
        '--depth-scale',
        type=float,
        default=1.0,
        help="Additional depth scaling factor to apply."
    )

    default_samples_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'samples'
    )
    group.add_argument(
        '--samples-path',
        type=str,
        default=default_samples_dir,
        help="Directory containing the samples."
    )
    group.add_argument(
        '--output-path',
        type=str,
        default=None,
        help="Directory to save the results."
    )
    group.add_argument(
        '--show-results',
        action='store_true',
        default=False,
        help="Show results in a window."
    )

    return parser.parse_args()


def _load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():
    args = _get_args()
    # assert all(x in args.input_modalities for x in ('rgb', 'depth', 'rgbd')), \
    #     "Only RGBD inference supported so far"

    device = torch.device(args.device)

    # data and model
    data = get_datahelper(args)
    dataset_config = data.dataset_config
    dataset_path_provided = bool(args.dataset_path)

    if not dataset_path_provided and not args.enable_linear_probing:
        raise ValueError(
            "Embedding predictions require either '--dataset-path' or "
            "'--enable-linear-probing'."
        )
    if args.enable_linear_probing:
        model = DVEFormerLinearProbing(args, dataset_config=dataset_config)
    else:
        model = DVEFormer(args, dataset_config=dataset_config)

    # load weights
    print(f"Loading checkpoint: '{args.weights_filepath}'")
    checkpoint = torch.load(args.weights_filepath,
                            map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    if 'epoch' in checkpoint:
        print(f"-> Epoch: {checkpoint['epoch']}")
    load_weights(args, model, state_dict, verbose=True)

    torch.set_grad_enabled(False)
    model.eval()
    model.to(device)

    # build preprocessor
    preprocessor = get_preprocessor(
        args,
        dataset=data.datasets_valid[0],
        phase='test',
        multiscale_downscales=None
    )

    # get samples
    basepath = args.samples_path
    # Files are assumed to be in an rgb and depth folder
    rgb_filepaths = sorted(glob(os.path.join(basepath, 'rgb', '*.*')))
    depth_filepaths = sorted(glob(os.path.join(basepath, 'depth', '*.*')))
    assert len(rgb_filepaths) == len(depth_filepaths)
    basenames_rgb = [os.path.basename(os.path.splitext(x)[0])
                     for x in rgb_filepaths]
    basenames_depth = [os.path.basename(os.path.splitext(x)[0])
                       for x in depth_filepaths]
    assert basenames_rgb == basenames_depth

    for fp_rgb, fp_depth in tqdm(zip(rgb_filepaths, depth_filepaths), total=len(rgb_filepaths)):
        # load rgb and depth image
        img_rgb = _load_img(fp_rgb)

        img_depth = _load_img(fp_depth).astype('float32')
        if args.depth_max is not None:
            img_depth[img_depth > args.depth_max] = 0
        img_depth *= args.depth_scale

        # preprocess sample
        sample = preprocessor({
            'rgb': img_rgb,
            'depth': img_depth,
            'identifier': os.path.basename(os.path.splitext(fp_rgb)[0])
        })

        # add batch axis as there is no dataloader
        batch = mt_collate([sample])
        batch = move_batch_to_device(batch, device=device)

        # apply model
        predictions = model(batch, do_postprocessing=True)
        # visualize predictions
        preview_map = {}
        prediction_visualizations = visualize_predictions(
            predictions=predictions,
            batch=batch,
            dataset_config=dataset_config
        )

        def _add_embedding_preview(slug: str, base_key: str, title: str) -> None:
            images = prediction_visualizations.get(base_key)
            if not images:
                return
            preview_map[slug] = (title, slug, images[0])

        _add_embedding_preview('text_based', 'dense_visual_embedding_text_based_semantic_idx', 'Text-based')
        _add_embedding_preview('visual_mean', 'dense_visual_embedding_visual_mean_based_semantic_idx', 'Visual Mean')
        _add_embedding_preview('linear_probing', 'linear_probing_semantic_idx', 'Linear Probing')

        if not preview_map:
            raise ValueError(
                'No embedding predictions available. Provide a dataset path or enable the linear probing head.'
            )

        ordered_slugs = ['text_based', 'visual_mean', 'linear_probing']
        previews = [
            preview_map[slug]
            for slug in ordered_slugs
            if slug in preview_map
        ]

        if args.output_path is not None:
            for title, slug, image in previews:
                if image is None:
                    continue
                fp_out = os.path.join(
                    args.output_path,
                    slug,
                    os.path.basename(fp_rgb)
                )
                # Create dir if not exists
                os.makedirs(os.path.dirname(fp_out), exist_ok=True)
                if isinstance(image, Image.Image):
                    image.save(fp_out)
                elif isinstance(image, np.ndarray):
                    # Convert to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(fp_out, image)

        if args.show_results:
            cols = 2 + len(previews)
            fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4), dpi=150)
            axes = np.atleast_1d(axes).tolist()
            for ax in axes:
                ax.set_axis_off()

            axes[0].set_title('RGB')
            axes[0].imshow(img_rgb)

            axes[1].set_title('Depth')
            axes[1].imshow(img_depth, interpolation='nearest')

            for ax, (title, slug, image) in zip(axes[2:], previews):
                if image is None:
                    continue
                ax.set_title(title)
                ax.imshow(np.asarray(image.convert('RGB')), interpolation='nearest')

            plt.tight_layout()
            if args.output_path is not None:
                plt.savefig(
                    os.path.join(
                        args.output_path, f"{os.path.basename(fp_rgb)}"
                    )
                )
            plt.show()


if __name__ == '__main__':
    main()
