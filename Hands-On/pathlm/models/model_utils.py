import logging
import os

import torch

from pathlm.evaluation.utility_metrics import NDCG, PRECISION, RECALL, MRR


class EarlyStopping:
    def __init__(self, patience, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time the monitored metric improved.
                            The training will stop if the metric does not improve for this number of epochs.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, ndcg_value):
        score = ndcg_value

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = 1
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch += self.counter + 1
            self.counter = 0

def save_model(model, model_dir, args, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = f'{model.name}_epoch_{current_epoch}_e{args.embed_dim}_bs{args.train_batch_size}_lr{args.lr}.pth'
    model_state_file = os.path.join(model_dir, filename)
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_filename = f'{model.name}_epoch_{last_best_epoch}_e{args.embed_dim}_bs{args.train_batch_size}_lr{args.lr}.pth'
        old_model_state_file = os.path.join(model_dir, old_filename)
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))


def load_model(model, model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def logging_metrics(epoch, metrics_dict, Ks):
    for k in Ks:
        if k not in metrics_dict:
            metrics_str = ', '.join([f'{key}: {metrics_dict[key]:.4f}' for key in metrics_dict.keys()])
            logging.info(f'Epoch {epoch} | K: {k} | {metrics_str}')
        else:
            #Log metric key metric value using join on dict
            metrics_str = ', '.join([f'{key}: {metrics_dict[k][key]:.4f}' for key in metrics_dict[k].keys()])
            logging.info(f'Epoch {epoch} | K: {k} | {metrics_str}')

def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):

    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".log")
    print("All logs will be saved to %s" %logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder
