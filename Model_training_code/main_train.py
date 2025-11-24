from dataset import Conv1Dataset
from model import ResNet1D
from train_eval_fn import train_model,test_model
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import os
import time
import joblib

GJ_coupling = 'strong'  # 'strong' or 'weak'
Load_model = False
full_seq = True
hidden_dims = [128, 128, 256, 256, 128]
batch_size = 512
num_workers = 32

scaler_name =f"{GJ_coupling}_standard_scaler.pkl"
if full_seq:
    best_model_state_name = f"best_ResNet_{GJ_coupling}_fullseq.pth"
else:
    best_model_state_name = f"best_ResNet_{GJ_coupling}.pth"
model_state_path = "Model_state/ResNet"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_DS = Conv1Dataset(device)
train_DS.normalize()
scaler = train_DS.stats
val_DS = Conv1Dataset(device=device,mode='validation',stats=scaler)
test_DS = Conv1Dataset(device=device,mode='test',stats=scaler)

test_DS.normalize()
val_DS.normalize()
train_DL = DataLoader(train_DS, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_DL = DataLoader(val_DS, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_DL = DataLoader(test_DS, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


mymodel = ResNet1D()

loss_fn = nn.MSELoss()
optimizer = optim.AdamW(mymodel.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)

joblib.dump(scaler, os.path.join("Scaler", scaler_name))
start_time = time.time()
if not Load_model:
    print(f"train new model {GJ_coupling}. Saving model into {best_model_state_name}")
    best_model_state = train_model(mymodel, 
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

avg_test_loss = test_model(mymodel,
                            test_DL,
                            loss_fn
                            )
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds. Test Average Loss: {avg_test_loss}")
