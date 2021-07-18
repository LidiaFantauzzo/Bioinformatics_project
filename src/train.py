import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


class Trainer:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args

        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay= args.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=args.lr_decay_step,gamma=args.lr_decay_factor)

        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def train(self, cur_epoch, train_loader, metrics=None, interval=5):
        """Train and return epoch loss"""
        device = self.device
        model = self.model
        optim = self.optimizer
        scheduler = self.scheduler

        epoch_losses = []
        interval_loss = 0.0

        model.train()
        for step, (images, labels) in enumerate(train_loader):

            images = images.to(device, dtype=torch.float32)#.unsqueeze(dim = 1)
            labels = labels.to(device, dtype=torch.long)

            optim.zero_grad()

            if self.args.model == 'bisenetv2' and self.args.output_aux:

                outputs, feat2, feat3, feat4, feat5_4 = self.model(images)

                loss = self.criterion(outputs, labels)
                loss2 = self.criterion(feat2, labels)
                loss3 = self.criterion(feat3, labels)
                loss4 = self.criterion(feat4, labels)
                loss5_4 = self.criterion(feat5_4, labels)
                boost_loss = loss2 + loss3 + loss4 + loss5_4

                loss_tot = loss + boost_loss

            else:
                outputs = self.model(images)
                # boost_loss = None
                loss_tot = self.criterion(outputs, labels)

            loss_tot.backward()

            optim.step()
            scheduler.step()

            interval_loss += loss_tot.item()

            _, prediction = outputs.max(dim=1)  # B, H, W
            labels = labels.cpu().numpy()
            prediction = prediction.cpu().numpy()
            if metrics is not None:
                metrics.update(labels, prediction)

            if (step + 1) % interval == 0:
                interval_loss = interval_loss / interval
                epoch_losses.append(interval_loss)
                print(f"Epoch {cur_epoch}, Batch {step + 1}/{len(train_loader)}," f" Loss={interval_loss}")

                interval_loss = 0.0

        return epoch_losses
        
    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def validate(self, loader, metrics, plt_samples_bool=False, test = False):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        model.eval()
        if self.args.dropout and test:
            self.enable_dropout(model)

        class_loss = 0.0

        plt_samples = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                outputs = self.model(images, test=True) if self.args.model == 'bisenetv2' else self.model(images)

                loss = self.criterion(outputs, labels)

                class_loss += loss.item()

                _, prediction = outputs.max(dim=1)  # B, H, W
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if plt_samples_bool:  # get samples
                    plt_samples.append((images[30].detach().cpu().numpy(),
                                        labels[30], prediction[30]))

        return class_loss, plt_samples