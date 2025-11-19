import os
import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, model='LSTM',file_name=None):
        if file_name is None:
            file_name = "Best_model.pth"
        self.file_name = file_name
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        self.model = model
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(root_dir, "Model_state", model)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            torch.save(self.best_model_state, os.path.join(self.model_path, self.file_name))
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
            torch.save(self.best_model_state, os.path.join(self.model_path, self.file_name))
