import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from models.Generator import Generator
from models.Discriminator import Discriminator
from Utilities.Save import load_checkpoint, save_checkpoint, save_examples
from Utilities.Data import MapDataset
from Utilities.Zip import zip_files_in_dir
import wandb
wandb.init(project="P2P_GAN")

torch.backends.cudnn.benchmark = True


# Hyperparameters
def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--val-dir", default="data/val")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--channels-img", type=int, default=3)
    parser.add_argument("--l1-lambda", type=int, default=100)
    parser.add_argument("--lambda-gp", type=int, default=10)
    parser.add_argument("--load-model", default=False)
    parser.add_argument("--save-model", default=False)
    parser.add_argument("--checkpoint-disc", default="disc.pth.tar")
    parser.add_argument("--checkpoint-gen", default="gen.pth.tar")

    return parser.parse_args()


def main():
    args = arguments()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")  # Can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, bce, d_scaler, g_scaler):
        loop = tqdm(loader, leave=True)

        for idx, (x, y) in enumerate(loop):
            x, y = x.to(device), y.to(device)

            # Train Discriminator
            with torch.cuda.amp.autocast():
                y_fake = gen(x)
                d_real = disc(x, y)
                d_fake = disc(x, y_fake.detach())
                d_real_loss = bce(d_real, torch.ones_like(d_real))
                d_fake_loss = bce(d_fake, torch.zeros_like(d_fake))
                d_loss = (d_real_loss + d_fake_loss) / 2  # /2 so discriminator train slower than the generator

            disc.zero_grad()
            d_scaler.scale(d_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Train Generator
            with torch.cuda.amp.autocast():
                d_fake = disc(x, y_fake)
                g_fake_loss = bce(d_fake, torch.ones_like(d_fake))
                l1 = l1_loss(y_fake, y) * args.l1_lambda
                g_loss = g_fake_loss + l1

            opt_gen.zero_grad()
            g_scaler.scale(g_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            if idx % 10 == 0:
                loop.set_postfix(
                    D_real=torch.sigmoid(d_real).mean().item(),
                    D_fake=torch.sigmoid(d_fake).mean().item(),
                )

    disc = Discriminator(in_channels=3).to(device)
    gen = Generator(in_channels=3).to(device)
    opt_disc = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    if args.load_model:
        load_checkpoint(args.checkpoint_gen, gen, opt_gen, args.lr)
        load_checkpoint(args.checkpoint_disc, disc, opt_disc, args.lr)

    train_dataset = MapDataset(root_dir="data/maps/train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir="data/maps/val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(args.epochs):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler)

        if args.save_model and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=args.checkpoint_gen)
            save_checkpoint(disc, opt_disc, filename=args.checkpoint_disc)

        save_examples(device, gen, val_loader, epoch, folder="evaluation")

    wandb.save('my_checkpoint.pth.tar')
    print(f"*** Creating zip archive from directory ***")
    zip_files_in_dir('evaluation', 'evaluation.zip', lambda name: 'png' in name)
    wandb.save('evaluation.zip')


if __name__ == "__main__":
    main()
