from collections import OrderedDict
import os
from configs import get_config, AttributeDict
import utils
from generator.dataset import DataFolder
from generator.model import VAE, PropDecoderWithMotif
import torch.optim


if __name__ == "__main__":
    config: AttributeDict = get_config()
    utils.set_seed(config['generator_rand_seed'])

    vocab_mof, vocab_y, vocab_x, vocab_atom = utils.set_vocab(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(config, vocab_x, vocab_mof, vocab_y, vocab_atom).to(device)
    model_state, _, _, _ = torch.load(config['best_model'])
    model.load_state_dict(model_state)
    model.eval()

    model_y = PropDecoderWithMotif(latent_dim=config['predict_latent_dim'], latent_size=config['predict_latent_size'],
                                   vocab_y=vocab_y, n_layers=config['n_layers'],
                                   act='relu', batchnorm=False,
                                   dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model_y.parameters(), lr=0.001)

    if config['load_model_y']:
        print('---------------- continuing -------------------------------------')
        model_y_state, optimizer_state, epoch_start = torch.load(config['best_model_y'])
        model_y.load_state_dict(model_y_state)
        optimizer.load_state_dict(optimizer_state)
    else:
        epoch_start = 0

    loss_history = {
        'train': [],
        'val': [],
        'val_r2': []
    }

    for epoch in range(epoch_start, config['predict_train_epoch']):
        train_dataset = DataFolder(config['tensors_train_prop'], config['train_batch_size'])
        test_dataset = DataFolder(config['tensors_test_prop'], config['train_batch_size'])

        y_true = OrderedDict()
        y_pred = OrderedDict()
        epoch_train_loss = 0
        epoch_test_loss = 0
        for i in range(len(config['col_y'])):
            y_true[i] = []
            y_pred[i] = []
        torch.cuda.empty_cache()
        model_y.train()
        for batch_num, batch in enumerate(train_dataset):

            torch.set_grad_enabled(False)
            root_vecs, tree_vecs, mol_batch = utils.get_vecs(model, batch, device)
            torch.set_grad_enabled(True)

            losses, outputs = model_y(root_vecs, tree_vecs, mol_batch['y'], mol_batch['y_mask'])
            loss = sum(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            for i in range(len(config['col_y'])):
                y_pred[i].extend(outputs[:, i].cpu().detach().numpy().reshape(-1))
                y_true[i].extend(mol_batch['y'][:, i].cpu().numpy())
        avg_train_loss = epoch_train_loss / batch_num
        print(f"Train  Epoch [{epoch + 1}/{config['predict_train_epoch']}], Avg Loss: {avg_train_loss}")
        loss_history['train'].append(avg_train_loss)
        utils.regression_statistics(y_true, y_pred, config['col_y'])
        ckpt = (model_y.state_dict(), optimizer.state_dict(), epoch)
        torch.save(ckpt, os.path.join(config['train_y_save_dir'], f"model.ckpt.{(epoch + 1)}"))
        # torch.save(ckpt, config['best_model_y'])

        print("-----------------------------test-----------------------------------")
        model_y.eval()
        with torch.no_grad():
            y_true = OrderedDict()
            y_pred = OrderedDict()
            for i in range(len(config['col_y'])):
                y_true[i] = []
                y_pred[i] = []
            for batch_num, batch in enumerate(test_dataset):
                root_vecs, tree_vecs, mol_batch = utils.get_vecs(model, batch, device)

                losses, outputs = model_y(root_vecs, tree_vecs, mol_batch['y'], mol_batch['y_mask'])

                epoch_test_loss += sum(losses).item()
                y_pred_scaler = model_y.scaler.inverse_transform(outputs.cpu().numpy())
                y_true_scaler = model_y.scaler.inverse_transform(mol_batch['y'].cpu().numpy())
                for i in range(len(config['col_y'])):
                    y_true[i].extend(y_true_scaler[:, i].reshape(-1))
                    y_pred[i].extend(y_pred_scaler[:, i].reshape(-1))
            avg_val_loss = epoch_test_loss / batch_num
            print(f"Test  Epoch [{epoch + 1}/{config['predict_train_epoch']}], Avg Loss: {avg_val_loss}")
            loss_history['val'].append(avg_val_loss)
            r2 = utils.regression_statistics(y_true, y_pred, config['col_y'])
            loss_history['val_r2'].append(r2['R2'][4])

        print("-----------------------------test-end--------------------------------")
    with open("loss_log" + ".txt", "w") as f:
        f.write("Epoch,Train Loss,Val Loss,R2_CO2\n")
        count = 0
        for epoch in range(epoch_start, config['predict_train_epoch']):
            f.write(f"{epoch + 1},{loss_history['train'][count]:.4f},"
                    f"{loss_history['val'][count]:.4f},{loss_history['val_r2'][count]:.4f}\n")
            count += 1
