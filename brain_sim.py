import os
import re
import json
import functools
import numpy as np
import multiprocessing

import torch
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.brain_transformation import ModelCommitment

from brainscore import score_model

from utils.config import get_config_parser
from resnet import resnet
from ViT import ViT

benchmark_list = {
    'v1': 'movshon.FreemanZiemba2013public.V1-pls',
    'v2': 'movshon.FreemanZiemba2013public.V2-pls',
    'v4': 'dicarlo.MajajHong2015public.V4-pls',
    'IT': 'dicarlo.MajajHong2015public.IT-pls',
}


if __name__ == '__main__':

    parser = get_config_parser()
    args = parser.parse_args()

    if args.model_config is not None:
        print(f'Loading model config from {args.model_config}')
        with open(args.model_config) as f:
            model_config = json.load(f)
    else:
        raise ValueError('Please provide a model config json')

    print('############################################')

    name = args.model_name
    model_config['num_classes'] = args.num_classes
    if args.model_type == "resnet":
        model = resnet(**model_config)
        pattern = re.compile(r'(.*maxpool.*|.*downsample.1|.*bn1.*|.*bn2.*|.*relu.*|.*avgpool.*|.*classifier.*)')
        layers = [name for name, _ in model.named_modules() if pattern.match(name)]

    else:
        model = ViT(**model_config)

    model.to(args.device)

    if args.load_checkpoint:
        checkpoint = torch.load(args.model_name, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.model_name, map_location=torch.device('cpu')))
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(
        identifier=f'{os.path.splitext(os.path.basename(args.model_name))[0]}', 
        model=model, 
        preprocessing=preprocessing
    )

    model = ModelCommitment(
        identifier=f'{os.path.splitext(os.path.basename(args.model_name))[0]}', 
        activations_model=activations_model,
        layers=layers
        )

    for region, benchmark in benchmark_list.items():
        score = score_model(model_identifier=model.identifier, model=model, benchmark_identifier=benchmark)
        np.save(os.path.join(args.logdir, f'{os.path.splitext(os.path.basename(args.model_name))[0]}.npy'), score.values)
