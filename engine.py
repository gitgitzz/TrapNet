
import os
import torch
import torch.nn.functional as F
from copy import deepcopy
import time
import tqdm
import kornia
import json

import losses as L


class LabelObfuscation:
    def __init__(self, args):
        self.attack_mode = args.attack_mode
        self.num_classes = args.num_classes
        self.device = args.device
        self.data_name = args.data_name
        self.label_matches = self.read_label_matches()

    def read_label_matches(self):
        with open(f"./data_files/{self.data_name}_node_matches_min.txt", "r") as fp:
            node_dict = json.load(fp)
        return node_dict

    def create_targets(self, targets):  # fake -> real
        if self.attack_mode == "random":
            bd_targets = torch.tensor([(label + 1) % self.num_classes for label in targets])
        elif self.attack_mode == "matching":
            bd_targets = torch.tensor([self.label_matches[str(int(label))] for label in targets])
        else:
            raise Exception("{} attack mode is not implemented".format(self.attack_mode))
        return bd_targets.type(torch.long).to(self.device)

    def reverse_targets(self, targets):  # real -> fake
        if self.attack_mode == "random":
            bd_targets = torch.tensor([(label - 1) % self.num_classes for label in targets])
        elif self.attack_mode == "matching":
            bd_targets = torch.tensor([self.label_matches[str(int(label))] for label in targets])
        else:
            raise Exception("{} attack mode is not implemented".format(self.attack_mode))
        return bd_targets.type(torch.long).to(self.device)


def train(args, model, optimizer, trainloader, epoch=0):
    device = torch.device('cuda')
    # adjust_epochs, gamma = [25, 35], 0.2
    adjust_epochs, gamma = [10, 20], 0.1  # for vgg
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=adjust_epochs, gamma=gamma)

    tf = time.time()
    ACC, cnt, loss_tot = 0, 0, 0.0
    entropy, loss1, loss2, loss3, loss4 = 0., .0, .0, .0, .0
    model.train()
    for img, iden in tqdm.tqdm(trainloader, leave=False, desc="Training"):
        img, iden = img.to(device), iden.to(device)
        bs = img.size(0)
        iden = iden.view(-1)

        if args.use_smooth_y == True:
            feats, out_prob = model(img)
            loss = L.smooth_loss(out_prob, iden)
        else:
            feats, out_prob = model(img)
            loss = L.cross_entropy_loss(out_prob, iden)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        out_iden = torch.argmax(out_prob, dim=1).view(-1)
        ACC += torch.sum(iden == out_iden).item()
        loss_tot += loss.item() * bs
        cnt += bs
        with torch.no_grad():
            entropy += L.cross_entropy_loss(out_prob, F.softmax(out_prob, dim=1)) * bs
            loss1 += (L.cross_entropy_loss(out_prob, iden) - L.cross_entropy_loss(out_prob, F.softmax(out_prob, dim=1))) *bs
            loss2 += (L.cross_entropy_loss(out_prob, iden) - F.log_softmax(out_prob, dim=1).mean()) * 0.5 *bs
            loss3 += torch.nn.CrossEntropyLoss(label_smoothing=0.5)(out_prob, iden) *bs
            loss4 += L.max_margin_loss(out_prob, iden) *bs

    train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt

    interval = time.time() - tf
    if epoch % 10 == 0:
        torch.save({'state_dict': model.state_dict(),
                    'opt': optimizer.state_dict(),
                    },
                   os.path.join(args.results_root, "allclass_epoch{:03d}.tar").format(epoch)
                   )

    print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tcurrent lr:{:.2f}".format(epoch, interval, train_loss, train_acc, scheduler.get_last_lr()[-1]))
    print("\tH(q):{:.2f}\tloss1:{:0.2f}\tloss2:{:0.2f}\tloss3:{:0.2f}\tloss4:{:0.2f}".format(entropy/cnt, loss1/cnt, loss2/cnt, loss3/cnt, loss4/cnt))

    scheduler.step()


def test(args, model, dataloader, epoch, best_ACC=0.):
    criterion = torch.nn.CrossEntropyLoss()
    device = args.device
    model.eval()
    loss, cnt, ACC = 0.0, 0, 0
    entropy, loss1, loss2, loss3, loss4 = 0., .0, .0, .0, .0
    with torch.no_grad():
        for img, iden in tqdm.tqdm(dataloader, leave=False, desc="testing"):
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)

            out_prob = model(img)[-1]
            loss += criterion(out_prob, iden).item() * bs
            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            cnt += bs
            entropy += L.cross_entropy_loss(out_prob, F.softmax(out_prob, dim=1)) * bs
            loss1 += (L.cross_entropy_loss(out_prob, iden) - L.cross_entropy_loss(out_prob, F.softmax(out_prob, dim=1))) *bs
            loss2 += (L.cross_entropy_loss(out_prob, iden) - F.log_softmax(out_prob, dim=1).mean()) * 0.5 *bs
            loss3 += torch.nn.CrossEntropyLoss(label_smoothing=0.5)(out_prob, iden) *bs
            loss4 += L.max_margin_loss(out_prob, iden) *bs

    test_acc = ACC * 100.0 / cnt
    test_loss = loss / cnt

    print("Test:\tBest Acc:{:.2f}\tTest Acc:{:.2f}\tTest Loss:{:.2f}".format(best_ACC, test_acc, test_loss))
    print("\tH(q):{:.2f}\tloss1:{:0.2f}\tloss2:{:0.2f}\tloss3:{:0.2f}\tloss4:{:0.2f}".format(entropy/cnt, loss1/cnt, loss2/cnt, loss3/cnt, loss4/cnt))
    if test_acc > best_ACC:
        best_ACC = test_acc
        best_model = deepcopy(model)
        torch.save({'state_dict': best_model.state_dict()}, os.path.join(args.results_root, "allclass_best.tar"))
        print("save ckpt at epoch {} with best acc={:.2f}".format(epoch, best_ACC))

    model.train()
    return best_ACC, test_acc
