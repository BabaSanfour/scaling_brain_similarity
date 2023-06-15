from brainscore.benchmarks import public_benchmark_pool
from config import get_config_parser
import torch
import os
import json
from resnet import resnet
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
import functools
from model_tools.brain_transformation import ModelCommitment
from brainscore import score_model

if __name__ == '__main__':

    parser = get_config_parser()
    args = parser.parse_args()

    if args.model_config is not None:
        print(f'Loading model config from {args.model_config}')
        with open(args.model_config) as f:
            model_config = json.load(f)
    else:
        raise ValueError('Please provide a model config json')
    name = args.model_name
    benchmark = public_benchmark_pool['dicarlo.MajajHong2015public.IT-pls']
    model = resnet(**model_config)
    model.load_state_dict(torch.load(args.model_name, map_location=torch.device('cpu')))
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier=f'{os.path.basename(args.model_config)}_{args.scaling_factor}', model=model, preprocessing=preprocessing)
    model = ModelCommitment(identifier=f'{os.path.basename(args.model_config)}_{args.scaling_factor}', activations_model=activations_model,
                        # specify layers to consider
                        layers=[ 'model.block3.4.bn2', 'model.block3.4.relu', 'model.block3.5.bn1', 'model.block3.5.bn2',  'model.block3.5.relu',
                    'model.block4.0.bn1',  'model.block4.0.bn2', 'model.block4.0.relu', 'model.block4.1.bn1', 'model.block4.1.bn2', 'model.block4.1.relu',  'model.block4.2.bn1',  'model.block4.2.bn2',
                    'model.block4.2.relu'])
    score = score_model(model_identifier=model.identifier, model=model,
                    benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
    print(score)
    with open(f"/home/hamza97/scratch/scaling_net_weights/{os.path.basename(args.model_config)}_{args.scaling_factor}.json", "w") as outfile:
        outfile.write(score)
