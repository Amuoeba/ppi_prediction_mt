# General imports
import os
from collections import OrderedDict
import time
# Project specific imports
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# Imports from internal libraries
import config
from datasets.distogram_sequence_dataset import DistogramSequenceDataset, CachedDistogramSequenceDataset, PDBindDataset, PandasMolStructure
import utils
from nn_utils.hook_manager import HookManager, VisHookManager, VisDispatcher


# Typing imports
from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class InteractionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.downscale = nn.Sequential(
            OrderedDict([
                ("cnn_1", nn.Conv2d(6, 12, 3, stride=1, padding=1)),
                ("batchnorm_1", nn.BatchNorm2d(12)),
                ("relu_1", nn.ReLU(inplace=True)),

                ("cnn_2", nn.Conv2d(12, 24, 3, stride=1, padding=1)),
                ("batchnorm_2", nn.BatchNorm2d(24)),
                ("relu_2", nn.ReLU(inplace=True)),

                ("cnn_3", nn.Conv2d(24, 48, 3, stride=1, padding=1)),
                ("batchnorm_3", nn.BatchNorm2d(48)),
                ("relu_3", nn.ReLU(inplace=True)),

                ("cnn_4", nn.Conv2d(48, 48, 3, stride=2, padding=1)),
                ("batchnorm_4", nn.BatchNorm2d(48)),
                ("relu_4", nn.ReLU(inplace=True)),

                ("cnn_5", nn.Conv2d(48, 48, 3, stride=2, padding=1)),
                ("batchnorm_5", nn.BatchNorm2d(48)),
                ("relu_5", nn.ReLU(inplace=True)),

                ("cnn_6", nn.Conv2d(48, 96, 3, stride=2, padding=1)),
                ("batchnorm_6", nn.BatchNorm2d(96)),
                ("relu_6", nn.ReLU(inplace=True)),

                ("cnn_7", nn.Conv2d(96, 192, 3, stride=2, padding=1)),
                ("batchnorm_7", nn.BatchNorm2d(192)),
                ("relu_7", nn.ReLU(inplace=True))
            ])
        )
        self.T_cnn_1 = nn.ConvTranspose2d(
            192, 96, 3, stride=2, padding=1, bias=False)
        self.Ubatchnorm_1 = nn.BatchNorm2d(96)
        self.U_relu_1 = nn.ReLU(True)

        self.T_cnn_2 = nn.ConvTranspose2d(
            96, 48, 3, stride=2, padding=1, bias=False)
        self.Ubatchnorm_2 = nn.BatchNorm2d(48)
        self.U_relu_2 = nn.ReLU(True)

        self.T_cnn_3 = nn.ConvTranspose2d(
            48, 24, 3, stride=2, padding=1, bias=False)
        self.Ubatchnorm_3 = nn.BatchNorm2d(24)
        self.U_relu_3 = nn.ReLU(True)

        self.T_cnn_4 = nn.ConvTranspose2d(
            24, 1, 3, stride=2, padding=1, bias=False)
        self.U_sigmoid_out = nn.Sigmoid()

    def forward(self, input):
        x = self.downscale(input)
        x_size = torch.tensor(x.size()) * 2
        x = self.T_cnn_1(x, output_size=x_size)
        x = self.Ubatchnorm_1(x)
        x = self.U_relu_1(x)

        x_size = torch.tensor(x.size()) * 2
        x = self.T_cnn_2(x, output_size=x_size)
        x = self.Ubatchnorm_2(x)
        x = self.U_relu_2(x)

        x_size = torch.tensor(x.size()) * 2
        x = self.T_cnn_3(x, output_size=x_size)
        x = self.Ubatchnorm_3(x)
        x = self.U_relu_3(x)

        x_size = torch.tensor(x.size()) * 2
        x = self.T_cnn_4(x, output_size=x_size)
        x = self.U_sigmoid_out(x)
        return x

    def __repr__(self):
        s = f"{'-'*20}{'MODEL':^20}{'-'*20}\n"
        s += super().__repr__() + "\n"
        s += f"{'-'*60}"
        return s


def weights_init(m):
    """Weight initialization function that was used in DCGAN paper.
    Should be used by applying it to nn.Module
    Args:
        m: Module to apply it to
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")
    print(f"Process id.: {os.getpid()}")
    # Define the training data
    # sql_db = PDBindDataset(config.PDBIND_SQLITE_DB)
    # samples = sql_db.get_2chain_samples()
    # train_set = DistogramSequenceDataset(
    #     samples, 512, set_type="train", feature_type="stacked")

    # Cached training data if exists
    cache_root = config.model_one_cfg.DATALOADER_CACHE
    train_set = CachedDistogramSequenceDataset(
        cache_root, "train", split_variant=0)
    val_set = CachedDistogramSequenceDataset(
        cache_root, "val", split_variant=0)
    test_set = CachedDistogramSequenceDataset(
        cache_root, "test", split_variant=0)

    train_loader = DataLoader(train_set, batch_size=config.model_one_cfg.BATCH_SIZE,
                              shuffle=True, num_workers=config.model_one_cfg.DATALOADER_WORKERS,
                              drop_last=True)

    val_loader = DataLoader(val_set, batch_size=3,
                            shuffle=False, num_workers=10,
                            drop_last=True)

    # Device selection
    ngpu = 1
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(f"Using device: {device}")

    # Model
    interactionNet = InteractionNet().to(device)
    interactionNet.apply(weights_init)

    # Initialize hook managers
    hook_manager = HookManager(interactionNet,
                               [{"downscale": ["cnn_1", "cnn_2", "cnn_3", "cnn_4", "cnn_5", "cnn_6", "cnn_7"]},
                                "T_cnn_1", "T_cnn_2", "T_cnn_3", "T_cnn_4"])
    # hook_manager.register_hooks()

    vis_dispatcher = VisDispatcher(config.model_one_cfg.VIS_WORKERS_ACT, config.model_one_cfg.VIS_WORKERS_FILT)
    vis_hook_manager = VisHookManager(interactionNet, vis_dispatcher,
                                      activation_vis_hookable=[{"downscale": ["cnn_1"]}],
                                      filter_vis_hookable=[{"downscale": ["cnn_1", "cnn_2", "cnn_3", "cnn_4", "cnn_5", "cnn_6", "cnn_7"]},
                                                           "T_cnn_1", "T_cnn_2", "T_cnn_3", "T_cnn_4"])

    # Initialize logger
    utils.logger.init()

    print(interactionNet)
    print(
        f"Number of trainable params: {utils.get_num_params(interactionNet)}")

    # Loss function
    loss_fn = nn.MSELoss()

    # Optimizer1
    optimizer = optim.Adam(interactionNet.parameters(),
                           lr=0.001, eps=1e-08, weight_decay=0.01)

    epochs = 250
    print("Training started")
    for e in range(epochs):
        interactionNet.train()
        # Activate vis_hooks at new epoch entrance
        # vis_hook_manager.register_activation_hooks()
        # vis_hook_manager.register_filter_hooks()
        for i, sample_batch in enumerate(train_loader):

            # Read batch and transfer tensors to appropriate device
            batch_files = sample_batch["pdb_path"]

            # Coment below lines to select wich features are used in training
            in_features = sample_batch["feature"].to(device, dtype=torch.float)
            # in_features = sample_batch["feature"][:,[0,3],:,:].to(device, dtype=torch.float)

            target = sample_batch["dg_inter"].to(device, dtype=torch.float)
            # Put gradinets to 0
            interactionNet.zero_grad()

            output = interactionNet(in_features)
            loss = loss_fn(output, target)
            print(f"Epoch: {e}, Batch: {i}, Loss: {loss}")
            utils.logger.log_training(e, i, loss)
            loss.backward()
            optimizer.step()

            # Terminate vis_hooks after first epoch batch
            # vis_hook_manager.unregister_activation_vis_hooks()
            # vis_hook_manager.unregister_filter_vis_hooks()

            if i % 30 == 0:
                log_out = output.detach().cpu().numpy()[0][0]
                log_target = target.detach().cpu().numpy()[0][0]
                utils.logger.save_target_VS_output(
                    log_target, log_out, "test", e, i)

            # Uncoment for registering debug hooks
            # if i % 30 == 0:
            #     hook_manager.register_hooks()
            # else:
            #     hook_manager.unregister_hooks()
            # TODO Create bach samplers and visualizers. Set up VISDOM

            a = 1

        print(f"{' Validating ':-^60}")
        with torch.no_grad():
            vis_hook_manager.register_activation_hooks()
            vis_hook_manager.register_filter_hooks()
            interactionNet.eval()
            for i, sample_batch in enumerate(val_loader):

                in_features = sample_batch["feature"].to(
                    device, dtype=torch.float)
                target = sample_batch["dg_inter"].to(device, dtype=torch.float)

                val_batch_files = sample_batch["pdb_path"]
                vis_hook_manager.set_run_metadata(e, val_batch_files)

                output = interactionNet(in_features)
                loss = loss_fn(output, target)
                print(f"Batch: {i}, Loss: {loss}")
                utils.logger.log_validation(e,i, loss)
                vis_hook_manager.unregister_activation_vis_hooks()
                vis_hook_manager.unregister_filter_vis_hooks()
        print("-"*60)


count_sleep = 0
while not vis_dispatcher.activations_vis_queue.empty():
    count_sleep += 1
    # py_proc = psutil.Process(os.getpid())
    # mem_use = py_proc.memory_info()
    print(
        f"\rWaiting for que to be empty: {count_sleep}. Que len: {vis_dispatcher.activations_vis_queue.qsize()}", end="")
    # print(f"Current cpu usage {mem_use}")
    time.sleep(2)

vis_dispatcher.terminate_dispatcher()
