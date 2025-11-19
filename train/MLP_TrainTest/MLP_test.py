import torch

def test_MLP_autoregression(model,
                            test_ds,
                            device,
                            loss_fn
                            ):
    model.eval()     
    model.to(device)
    total_loss = 0.0
    pre_results = []
    post_results = []
    N = len(test_ds)
    with torch.no_grad():
        for i in range(N):
            x, y_true = test_ds[i]
            x = x.unsqueeze(0).to(device)  # Add batch dimension
            y_true = y_true.unsqueeze(0).to(device)

            y_pred = model(x)

            loss = loss_fn(y_pred, y_true)
            total_loss += loss.item()

            pre_results.append(y_pred[:,0].cpu().numpy())
            post_results.append(y_pred[:,1].cpu().numpy())
    avg_loss = total_loss / N
    return avg_loss, (pre_results, post_results)