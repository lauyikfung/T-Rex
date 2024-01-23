import sys
sys.path.append('TextRetrosynthesis/text_datasets/')

import os
import logging
import argparse
from torchdrug import data, utils
from dataset import USPTO50k
import torch
from torchdrug import core, models, tasks, metrics
from torchdrug.layers import functional
from torch.utils import data as torch_data
# import tensorboard
from tensorboardX import SummaryWriter


class newCenterTask(tasks.CenterIdentification):
    def __init__(self, model, feature):
        super(newCenterTask, self).__init__(model, feature)

    def variadic_top_precision(self, pred, target, size, k):
        index = functional.variadic_topk(pred, size, k, largest=True)[1]
        index = index.reshape([-1]) + (size.cumsum(0) - size).repeat_interleave(k)
        k_size = torch.tensor([k]*len(size)).to(size.device)
        precision = functional.variadic_sum(target[index], k_size)
        precision[size < k] = 0
        return torch.mean(precision.float())

    def evaluate(self, pred, target):
        target, size = target

        metric = {}
        target_vm = functional.variadic_max(target, size)[1]
        accuracy = metrics.variadic_accuracy(pred, target_vm, size).mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        for k in [1, 2, 3, 5]:
            p = self.variadic_top_precision(pred, target, size, k)
            name = tasks._get_metric_name(f"top:{k} precision")
            metric[name] = p

        return metric


def main(args):

    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
    else:
        pass
        # raise FileExistsError('The save directory already exists!')
    
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")
    handler = logging.FileHandler('{}/train.log'.format(args['save_dir']))
    handler.setFormatter(format)
    logger = logging.getLogger("")
    logger.addHandler(handler)

    if args['debug']:
        data_name = 'data_ChatGPT_debug.csv'
    else:
        data_name = 'data_ChatGPT_steps_test.csv'

    # set the random seed
    torch.manual_seed(args['fold'])
    # load the reaction dataset and the synthon dataset
    synthon_dataset = USPTO50k(args['data_dir'],
                                file_name=data_name,
                                atom_feature="synthon_completion", as_synthon=True,
                                with_hydrogen=False, kekulize=True, verbose=1)
    synthon_train, synthon_valid, synthon_test = synthon_dataset.split(ratios=[1, 0, 0])
    reaction_dataset = USPTO50k(args['data_dir'],
                                file_name=data_name,
                                atom_feature="center_identification", as_synthon=False,
                                with_hydrogen=False, kekulize=True, verbose=1)
    reaction_train, reaction_valid, reaction_test = reaction_dataset.split(ratios=[1, 0, 0])
    # create the reaction model
    reaction_model = models.RGCN(input_dim=reaction_dataset.node_feature_dim,
                    hidden_dims=[256, 256, 256, 256, 256, 256],
                    num_relation=reaction_dataset.num_bond_type,
                    concat_hidden=True)
    reaction_task = tasks.CenterIdentification(reaction_model,
                                           feature=("graph", "atom", "bond"))
    reaction_optimizer = torch.optim.Adam(reaction_task.parameters(), lr=args['reaction_lr'])
    print(args['device'])
    reaction_solver = core.Engine(reaction_task, reaction_train, reaction_valid,
                                reaction_test, reaction_optimizer,
                                gpus=[args['device']], batch_size=args['reaction_batch_size'],
                                log_interval=args['log_interval'])
  

    # initial the best metric
    reaction_solver.load(f"{args['save_dir']}/reaction_best.pth")
    reaction_task = reaction_solver.model
    reaction_metric = reaction_solver.evaluate("train")

    synthon_model = models.RGCN(input_dim=synthon_dataset.node_feature_dim,
                            hidden_dims=[256, 256, 256, 256, 256, 256],
                            num_relation=synthon_dataset.num_bond_type,
                            concat_hidden=True)
    synthon_task = tasks.SynthonCompletion(synthon_model, feature=("graph",))
    synthon_optimizer = torch.optim.Adam(synthon_task.parameters(), lr=args['synthon_lr'])
    synthon_solver = core.Engine(synthon_task, synthon_train, synthon_valid,
                                synthon_test, synthon_optimizer, gpus=[args['device']],
                                batch_size=args['synthon_batch_size'],
                                log_interval=args['log_interval'])

    # initial the best metric
    synthon_solver.load(f"{args['save_dir']}/synthon_best.pth")
    synthon_task = synthon_solver.model
    synthon_metric = synthon_solver.evaluate("train")

    task = tasks.Retrosynthesis(reaction_task, synthon_task, center_topk=2,
                            num_synthon_beam=5, max_prediction=10)
    lengths = [len(reaction_valid) // 10,
            len(reaction_valid) - len(reaction_valid) // 10]
    reaction_valid_small = torch_data.random_split(reaction_valid, lengths)[0]
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
    solver = core.Engine(task, reaction_train, reaction_valid_small, reaction_test,
                        optimizer, batch_size=args['retrosynthesis_batch_size'], gpus=[args['device']])
    solver.load(f"{args['save_dir']}/reaction_best.pth", load_optimizer=False)
    solver.load(f"{args['save_dir']}/synthon_best.pth", load_optimizer=False)
    val_metrics = solver.evaluate("train")

    f = open(os.path.join(args['save_dir'], 'retrosynthesis_result_test_new_split.txt'), 'w')
    f.write(f'\nreaction model result for fold {args["fold"]}\n')
    for key, value in reaction_metric.items():
        f.write(f'{key}: {value.cpu().numpy()}\n')
        f.write('\n')
    f.write(f'\nsynthon model result for fold {args["fold"]}\n')
    for key, value in synthon_metric.items():
        f.write(f'{key}: {value.cpu().numpy()}\n')
        f.write('\n')
    f.write(f'\nretrosynthesis result for fold {args["fold"]}\n')
    for key, value in val_metrics.items():
        f.write(f'{key}: {value.cpu().numpy()}\n')
    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--reaction_epoch', type=int, default=10)
    parser.add_argument('--synthon_epoch', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='data/candidate_generation/')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--reaction_lr', type=float, default=1e-3)
    parser.add_argument('--synthon_lr', type=float, default=1e-4)
    parser.add_argument('--reaction_batch_size', type=int, default=8)
    parser.add_argument('--synthon_batch_size', type=int, default=20)
    parser.add_argument('--retrosynthesis_batch_size', type=int, default=5)
    parser.add_argument('--best_metric_reaction', type=str, default='accuracy')
    parser.add_argument('--best_metric_synthon', type=str, default='total accuracy')
    parser.add_argument('--log_interval', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--device', type=int, default=0)
    
    args = parser.parse_args()
    args = args.__dict__
    main(args)

