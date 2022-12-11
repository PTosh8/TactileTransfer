import numpy as np
import os
import glob as glob
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset, ConcatDataset

from modules import VectorQuantizedVAE, to_scalar
from datasets import MiniImagenet, TactileTransfer

from tensorboardX import SummaryWriter

class ProcessedTactileTransfer(Dataset):
    def __init__(self, root_dir, split='train', bubbles_transform=None, gelslim_transform=None):
        """
        Args:
            root_dir: the directory of the dataset
            split: "train" or "val"
            transform: pytorch transformations.
        """

        self.bubbles_transform = bubbles_transform
        self.gelslim_transform = gelslim_transform
        self.bubbles_files = glob.glob(os.path.join(root_dir, 'processed_bubbles', split, '*.jpg'))
        self.gelslim_files = glob.glob(os.path.join(root_dir, 'processed_gelslim', split, '*.jpg'))

    def __len__(self):
        return len(self.bubbles_files)

    def __getitem__(self, idx):
        convert_tensor = transforms.ToTensor()

        bubbles_img = convert_tensor(Image.open(self.bubbles_files[idx]))
        gelslim_img = convert_tensor(Image.open(self.gelslim_files[idx]))

        if self.bubbles_transform:
            bubbles_img = self.bubbles_transform(bubbles_img)
        if self.gelslim_transform:
            gelslim_img = self.gelslim_transform(gelslim_img)

        return gelslim_img, bubbles_img

def train(data_loader, model, optimizer, args, writer):
    for images, labels in data_loader:
        images = images.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, labels)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + args.beta * loss_commit
        loss.backward()

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)

        optimizer.step()
        args.steps += 1

    return loss_recons.item(), loss_vq.item()

def test(data_loader, model, args, writer):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, labels in data_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, labels)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    # Logs
    writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
    writer.add_scalar('loss/test/quantization', loss_vq.item(), args.steps)

    return loss_recons.item(), loss_vq.item()

def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde

def main(args):
    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_filename = './models/{0}'.format(args.output_folder)

    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if args.dataset == 'mnist':
            # Define the train & test datasets
            train_dataset = datasets.MNIST(args.data_folder, train=True,
                download=True, transform=transform)
            test_dataset = datasets.MNIST(args.data_folder, train=False,
                transform=transform)
            num_channels = 1
        elif args.dataset == 'fashion-mnist':
            # Define the train & test datasets
            train_dataset = datasets.FashionMNIST(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(args.data_folder,
                train=False, transform=transform)
            num_channels = 1
        elif args.dataset == 'cifar10':
            # Define the train & test datasets
            train_dataset = datasets.CIFAR10(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(args.data_folder,
                train=False, transform=transform)
            num_channels = 3
        valid_dataset = test_dataset
    elif args.dataset == 'miniimagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(args.data_folder, train=True,
            download=True, transform=transform)
        valid_dataset = MiniImagenet(args.data_folder, valid=True,
            download=True, transform=transform)
        test_dataset = MiniImagenet(args.data_folder, test=True,
            download=True, transform=transform)
        num_channels = 3
    elif args.dataset == 'tactiletransfer':
        train_dataset = ConcatDataset([ProcessedTactileTransfer(os.path.join(args.data_folder, 'cylinder'), split='train'), ProcessedTactileTransfer(os.path.join(args.data_folder, 'cone'), split='train')])
        valid_dataset = ConcatDataset([ProcessedTactileTransfer(os.path.join(args.data_folder, 'cylinder'), split='val'), ProcessedTactileTransfer(os.path.join(args.data_folder, 'cone'), split='val')])
        test_dataset = ConcatDataset([ProcessedTactileTransfer(os.path.join(args.data_folder, 'cylinder'), split='test'), ProcessedTactileTransfer(os.path.join(args.data_folder, 'cone'), split='test')])
        num_channels = 3

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=2, shuffle=True)

    # Fixed images for Tensorboard
    interesting_inputs = torch.cat([test_dataset[1][0].unsqueeze(0), test_dataset[70][0].unsqueeze(0), test_dataset[59][0].unsqueeze(0), test_dataset[76][0].unsqueeze(0)], dim=0)
    interesting_gt = torch.cat([test_dataset[1][1].unsqueeze(0), test_dataset[70][1].unsqueeze(0), test_dataset[59][1].unsqueeze(0), test_dataset[76][1].unsqueeze(0)], dim=0)
    interesting_inputs_grid = make_grid(interesting_inputs.cpu(), nrow=2, normalize=True)
    interesting_gt_grid = make_grid(interesting_gt.cpu(), nrow=2, normalize=True)
    
    pic_getter = iter(train_loader)
    gelslims, bubbles = next(pic_getter)
    fixed_images = gelslims[50,:,:,:].unsqueeze(0)
    fixed_gt = bubbles[50,:,:,:].unsqueeze(0)
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).to(args.device)
    # with open('{0}/best.pt'.format(save_filename), 'rb') as f:
    #     model.load_state_dict(torch.load(f))
    #     model.eval()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, args)
    grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('reconstruction', grid, 0)

    best_loss = -1.
    print('Training...')
    train_recons_loss = np.empty(args.num_epochs)
    train_vq_loss = np.empty(args.num_epochs)
    test_recons_loss = np.empty(args.num_epochs)
    test_vq_loss = np.empty(args.num_epochs)
    for epoch in range(args.num_epochs):
        print(f"Epoch = {epoch}/{args.num_epochs}")
        train_recons, train_vq = train(iter(train_loader), model, optimizer, args, writer)
        train_recons_loss[epoch] = train_recons
        train_vq_loss[epoch] = train_vq

        test_recons, test_vq = test(iter(valid_loader), model, args, writer)
        test_recons_loss[epoch] = test_recons
        test_vq_loss[epoch] = test_vq

        loss = test_recons
        # reconstruction = generate_samples(fixed_images, model, args)
        # grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        # writer.add_image('reconstruction', grid, epoch + 1)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(model.state_dict(), f)

    dir = 'figures'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.2)
    ((ax1, ax2), (ax3, ax4)) = gs.subplots(sharex=True, sharey='row')
    

    ax1.set_yscale('log')
    ax1.plot(train_recons_loss)
    ax1.set(xlabel='Epochs', ylabel='Reconstruction Loss')
    ax1.set_title('Training data')

    ax2.set_yscale('log')
    ax2.plot(test_recons_loss)
    ax2.set(xlabel='Epochs', ylabel='Reconstruction Loss')
    ax2.set_title('Testing data')

    ax3.plot(train_vq_loss)
    ax3.set(xlabel='Epochs', ylabel='VQ Loss')
    ax3.set_title('Training data')

    ax4.plot(test_vq_loss)
    ax4.set(xlabel='Epochs', ylabel='VQ Loss')
    ax4.set_title('Testing data')

    fig.savefig(os.path.join(dir, 'losses.png'), bbox_inches='tight')


    interesting_outputs = generate_samples(interesting_inputs, model, args)
    interesting_outputs_grid = make_grid(interesting_outputs, nrow=2, normalize=True)

    if os.path.exists('sample_reconstruction.png'):
        os.remove('sample_reconstruction.png')
    save_image(interesting_outputs_grid, 'sample_reconstruction.png')
    
    if os.path.exists('input_image.png'):
        os.remove('input_image.png')
    save_image(interesting_inputs_grid, 'input_image.png')

    if os.path.exists('ground_truth.png'):
        os.remove('ground_truth.png')
    save_image(interesting_gt_grid, 'ground_truth.png')

    # input_image = (fixed_images[0,:,:,:].permute(1,2,0).numpy() * 255).astype(np.uint8)
    # sample_reconstruction = (reconstruction[0,:,:,:].to('cpu').permute(1,2,0).numpy() * 255).astype(np.uint8)
    
    # im = Image.fromarray(sample_reconstruction)
    # im.save('sample_reconstruction.jpg')
    # im = Image.fromarray(input_image)
    # im.save('input_image.jpg')



if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str,
        help='name of the data folder')
    parser.add_argument('--dataset', type=str,
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='vqvae',
        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./models/{0}'.format(args.output_folder)):
        os.makedirs('./models/{0}'.format(args.output_folder))
    args.steps = 0

    main(args)
