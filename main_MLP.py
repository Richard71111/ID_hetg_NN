import dataset
import model
import train.MLP_TrainTest as MLP_fn
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import os
import time

GJ_coupling = 'weak'
Load_model = False
full_seq = True
hidden_dims = [128, 128, 256, 256, 128]
batch_size = 512
num_workers = 4

if full_seq:
    best_model_state_name = f"best_mlp_{GJ_coupling}_fullseq.pth"
else:
    best_model_state_name = f"best_mlp_{GJ_coupling}.pth"
model_state_path = "Model_state/MLP"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_DS = dataset.MLPDataset(GJ_coupling=GJ_coupling, split='train',full_seq=full_seq)
val_DS = dataset.MLPDataset(GJ_coupling=GJ_coupling, split='val',full_seq=full_seq,scaler=train_DS.scaler)

train_DL = DataLoader(train_DS, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_DL = DataLoader(val_DS, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
input_features, output_features = train_DS.get_IO_features()
mymodel = model.MLP(input_features, hidden_dims, output_features, dropout=0.1)

loss_fn = nn.MSELoss()
optimizer = optim.AdamW(mymodel.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)

start_time = time.time()
if not Load_model:
    print(f"train new model {GJ_coupling}. Saving model into {best_model_state_name}")
    best_model_state = MLP_fn.train_MLP(mymodel, 
                                train_DL, 
                                val_DL, 
                                optimizer,
                                device,
                                epochs=100, 
                                scheduler=scheduler,
                                loss_fn=loss_fn,
                                patience = 10,
                                best_model_state_name = best_model_state_name
                                )
else:
    mymodel.load_state_dict(torch.load(os.path.join(model_state_path, best_model_state_name)))
    print("Loaded pre-trained model.")

test_DS = dataset.MLPDataset(GJ_coupling=GJ_coupling, split='test',full_seq=full_seq,scaler=train_DS.scaler)
avg_loss, results = MLP_fn.test_MLP_autoregression(mymodel,
                                                    test_DS,
                                                    device,
                                                    loss_fn
                                                    )
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds. Test Autoregression Average Loss: {avg_loss}")
