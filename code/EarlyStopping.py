class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=3, verbose=True, delta=1e-3, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, curr_val, model):

        if self.best_score is None:
            self.best_score = curr_val
            self.save_checkpoint(curr_val)
        elif curr_val > self.best_score:
            self.best_score = curr_val
            self.save_checkpoint(curr_val)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, curr_val):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation score improved ({self.val_score_max:.6f} --> {curr_val:.6f}).  Saving model ...')
        self.val_score_max = curr_val