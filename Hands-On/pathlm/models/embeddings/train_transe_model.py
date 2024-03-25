import os
import argparse
import torch
import torch.optim as optim

from pathlm.datasets.kg_dataset_base import KARSDataset
from pathlm.knowledge_graphs.kg_macros import ML1M, LFM1M, CELL
from pathlm.models.embeddings.kge_data_loader import KGEDataLoader
from pathlm.models.embeddings.kge_utils import get_log_dir, get_logger, get_embedding_ckpt_rootdir
from pathlm.models.embeddings.transe_model import TransE
from pathlm.utils import set_seed, SEED

logger = None


def train(args, dataset):
    dataloader = KGEDataLoader(dataset, args.batch_size)
    review_to_train = len(dataset.review.data) * args.epochs + 1
    EMBEDDING_CKPT_DIR = get_embedding_ckpt_rootdir(args.dataset)
    model = TransE(args, dataloader).to(args.device)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.SGD(model.parameters(), lr=args.lr)#optim.SGD(model.parameters(), lr=args.lr)
    steps = 0
    smooth_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        while dataloader.has_next():
            # Set learning rate.
            lr = args.lr * max(1e-4, 1.0 - dataloader.finished_review_num / float(review_to_train))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Get training batch.
            batch_idxs = dataloader.get_batch()
            batch_idxs = torch.from_numpy(batch_idxs).to(args.device)

            # Train models.
            optimizer.zero_grad()
            train_loss = model(batch_idxs)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            smooth_loss += train_loss.item() / args.steps_per_checkpoint

            steps += 1
            if steps % args.steps_per_checkpoint == 0:
                logger.info('Epoch: {:02d} | '.format(epoch) +
                            'Review: {:d}/{:d} | '.format(dataloader.finished_review_num, review_to_train) +
                            'Lr: {:.5f} | '.format(lr) +
                            'Smooth loss: {:.5f}'.format(smooth_loss))
                smooth_loss = 0.0
        if epoch % 10 == 0 or epoch == 1:
            torch.save(model.state_dict(), f'{EMBEDDING_CKPT_DIR}/transe_model_sd_epoch_{epoch}.ckpt')

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML1M, help='One of {beauty, cd, cell, clothing}.')
    parser.add_argument('--name', type=str, default='transe', help='models name.')
    parser.add_argument('--seed', type=int, default=SEED, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for adam.')
    parser.add_argument('--l2_lambda', type=float, default=0, help='l2 lambda')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Clipping gradient.')
    parser.add_argument('--embed_size', type=int, default=100, help='knowledge embedding size.')
    parser.add_argument('--num_neg_samples', type=int, default=5, help='number of negative samples.')
    parser.add_argument('--steps_per_checkpoint', type=int, default=200, help='Number of steps for checkpoint.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else (( torch.device('mps') if hasattr(torch.backends,'mps') and torch.backends.mps.is_available()\
        else 'cpu')  )
    #args.device = 'cpu'
    log_dir = get_log_dir(args.dataset, args.name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    global logger
    logger = get_logger(log_dir + '/train_log.txt')
    logger.info(args)

    set_seed(args.seed, False)
    dataset = KARSDataset(args.dataset)  #load_dataset(args.dataset)
    model = train(args, dataset)
    model.extract_embeddings(dataset, args.epochs)
    

if __name__ == '__main__':
    main()
