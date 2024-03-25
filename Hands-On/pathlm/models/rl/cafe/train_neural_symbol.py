import os
import sys
import numpy as np
import logging
import logging.handlers
import torch
import torch.optim as optim
import time

from pathlm.utils import SEED, get_weight_dir, get_weight_ckpt_dir

from pathlm.models.embeddings.kge_utils import load_embed, TRANSE
from pathlm.models.rl.CAFE.data_utils import OnlinePathLoader
from pathlm.models.rl.CAFE.symbolic_model import create_symbolic_model
from pathlm.models.rl.CAFE.cafe_utils import *
from pathlm.models.model_utils import create_log_id, logging_config

from pathlm.models.wadb_utils import MetricsLogger
logger = None

def set_logger(logname):
    global logger
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def train(args):
    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Setup logging
    log_save_id = create_log_id(args.log_dir)
    logging_config(folder=args.log_dir, name=f'log{log_save_id}', no_console=False)
    logging.info(args)

    # Setup device (GPU/CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device

    dataloader = OnlinePathLoader(args.dataset, args.batch_size, topk=args.topk_candidates)
    metapaths = dataloader.kg.metapaths

    kg_embeds = load_embed(args.dataset, 'transe') if train else None
    model = create_symbolic_model(args, dataloader.kg, train=True, pretrain_embeds=kg_embeds)
    params = [name for name, param in model.named_parameters() if param.requires_grad]
    logging.info(f'Trainable parameters: {params}')
    logging.info('==================================')

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    total_steps = args.epochs * dataloader.total_steps
    steps = 0
    smooth_loss = []
    smooth_reg_loss = []
    smooth_rank_loss = []

    model.train()
    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        while dataloader.has_next():
            # Update learning rate
            lr = args.lr * max(1e-4, 1.0 - steps / total_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # pos_paths: [bs, path_len], neg_paths: [bs, n, path_len]
            mpid, pos_paths, neg_pids = dataloader.get_batch()
            pos_paths = torch.from_numpy(pos_paths).to(args.device)
            neg_pids = torch.from_numpy(neg_pids).to(args.device)

            optimizer.zero_grad()
            reg_loss, rank_loss = model(metapaths[mpid], pos_paths, neg_pids)
            train_loss = reg_loss + args.rank_weight * rank_loss
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            smooth_loss.append(train_loss.item())
            smooth_reg_loss.append(reg_loss.item())
            smooth_rank_loss.append(rank_loss.item())

            if steps % args.steps_per_checkpoint == 0:
                smooth_loss = np.mean(smooth_loss)
                smooth_reg_loss = np.mean(smooth_reg_loss)
                smooth_rank_loss = np.mean(smooth_rank_loss)
                logging.info('Epoch/Step: {:02d}/{:08d} | '.format(epoch, steps) +
                            'LR: {:.5f} | '.format(lr) +
                            'Smooth Loss: {:.5f} | '.format(smooth_loss) +
                            'Reg Loss: {:.5f} | '.format(smooth_reg_loss) +
                            'Rank Loss: {:.5f} | '.format(smooth_rank_loss))
                smooth_loss = []
                smooth_reg_loss = []
                smooth_rank_loss = []
            steps += 1

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), f'{args.weight_dir_ckpt}/symbolic_model_epoch{epoch}.ckpt')

    torch.save(model.state_dict(), f'{args.weight_dir}/symbolic_model_epoch{args.epochs}.ckpt')
def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
