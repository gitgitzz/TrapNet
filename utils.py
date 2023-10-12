import json
import numpy
import os
import shutil
import torch
import torchvision
from models import classifiers
from models import defense_models

class Dict2Args(object):
    """Dict-argparse object converter."""

    def __init__(self, dict_args):
        for key, value in dict_args.items():
            setattr(self, key, value)


def generate_images(gen, device, batch_size=64, dim_z=128, distribution=None,
                    num_classes=None, class_id=None):
    """Generate images.

    Priority: num_classes > class_id.

    Args:
        gen (nn.Module): generator.
        device (torch.device)
        batch_size (int)
        dim_z (int)
        distribution (str)
        num_classes (int, optional)
        class_id (int, optional)

    Returns:
        torch.tensor

    """

    z = sample_z(batch_size, dim_z, device, distribution)
    if num_classes is None and class_id is None:
        y = None
    elif num_classes is not None:
        y = sample_pseudo_labels(num_classes, batch_size, device)
    elif class_id is not None:
        y = torch.tensor([class_id] * batch_size, dtype=torch.long).to(device)
    else:
        y = None
    with torch.no_grad():
        fake = gen(z, y)

    return fake


def sample_z(batch_size, dim_z, device, distribution=None):
    """Sample random noises.

    Args:
        batch_size (int)
        dim_z (int)
        device (torch.device)
        distribution (str, optional): default is normal

    Returns:
        torch.FloatTensor or torch.cuda.FloatTensor

    """

    if distribution is None:
        distribution = 'normal'
    if distribution == 'normal':
        return torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).normal_()
    else:
        return torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).uniform_()


def sample_pseudo_labels(num_classes, batch_size, device):
    """Sample pseudo-labels.

    Args:
        num_classes (int): number of classes in the dataset.
        batch_size (int): size of mini-batch.
        device (torch.Device): For compatibility.

    Returns:
        ~torch.LongTensor or torch.cuda.LongTensor.

    """

    pseudo_labels = torch.from_numpy(
        numpy.random.randint(low=0, high=num_classes, size=(batch_size))
    )
    pseudo_labels = pseudo_labels.type(torch.long).to(device)
    return pseudo_labels


def save_images(n_iter, count, root, train_image_root, fake, real):
    """Save images (torch.tensor).

    Args:
        root (str)
        train_image_root (root)
        fake (torch.tensor)
        real (torch.tensor)

    """

    fake_path = os.path.join(
        train_image_root,
        'fake_{}_iter_{:07d}.png'.format(count, n_iter)
    )
    real_path = os.path.join(
        train_image_root,
        'real_{}_iter_{:07d}.png'.format(count, n_iter)
    )
    torchvision.utils.save_image(
        fake, fake_path, nrow=4, normalize=True, scale_each=True
    )
    shutil.copy(fake_path, os.path.join(root, 'fake_latest.png'))
    torchvision.utils.save_image(
        real, real_path, nrow=4, normalize=True, scale_each=True
    )
    shutil.copy(real_path, os.path.join(root, 'real_latest.png'))


def save_checkpoints(args, n_iter, count, gen, opt_gen, dis, opt_dis, cls=None, opt_cls=None):
    """Save checkpoints.

    Args:
        args (argparse object)
        n_iter (int)
        gen (nn.Module)
        opt_gen (torch.optim)
        dis (nn.Module)
        opt_dis (torch.optim)

    """

    count = n_iter // args.checkpoint_interval
    gen_dst = os.path.join(args.results_root, 'gen_{:04d}_iter_{:07d}.pth.tar'.format(count, n_iter))
    torch.save({'model': gen.state_dict(), 'opt': opt_gen.state_dict()}, gen_dst)
    shutil.copy(gen_dst, os.path.join(args.results_root, 'gen_latest.pth.tar'))
    dis_dst = os.path.join(args.results_root, 'dis_{:04d}_iter_{:07d}.pth.tar'.format(count, n_iter))
    torch.save({'model': dis.state_dict(), 'opt': opt_dis.state_dict()}, dis_dst)
    shutil.copy(dis_dst, os.path.join(args.results_root, 'dis_latest.pth.tar'))
    if cls is not None:
        cls_dst = os.path.join(args.results_root, 'cls_{:04d}_iter_{:07d}.pth.tar'.format(count, n_iter))
        torch.save({'model': cls.state_dict(), 'opt': opt_cls.state_dict()}, cls_dst)
        shutil.copy(cls_dst, os.path.join(args.results_root, 'cls_latest.pth.tar'))


def resume_from_args(args_path, gen_ckpt_path, dis_ckpt_path):
    """Load generator & discriminator with their optimizers from args.json.

    Args:
        args_path (str): Path to args.json
        gen_ckpt_path (str): Path to generator checkpoint or relative path
                             from args['results_root']
        dis_ckpt_path (str): Path to discriminator checkpoint or relative path
                             from args['results_root']

    Returns:
        gen, opt_dis
        dis, opt_dis

    """

    from models.generators import resnet64
    from models.discriminators import snresnet64

    with open(args_path) as f:
        args = json.load(f)
    conditional = args['cGAN']
    num_classes = args['num_classes'] if conditional else 0
    # Initialize generator
    gen = resnet64.ResNetGenerator(
        args['gen_num_features'], args['gen_dim_z'], args['gen_bottom_width'],
        num_classes=num_classes, distribution=args['gen_distribution']
    )
    opt_gen = torch.optim.Adam(
        gen.parameters(), args['lr'], (args['beta1'], args['beta2'])
    )
    # Initialize discriminator
    args['dis_arch_concat'] = False
    if args['dis_arch_concat']:
        dis = snresnet64.SNResNetConcatDiscriminator(
            args['dis_num_features'], num_classes, dim_emb=args['dis_emb']
        )
    else:
        dis = snresnet64.SNResNetProjectionDiscriminator(
            args['dis_num_features'], num_classes
        )
    opt_dis = torch.optim.Adam(
        dis.parameters(), args['lr'], (args['beta1'], args['beta2'])
    )
    if not os.path.exists(gen_ckpt_path):
        gen_ckpt_path = os.path.join(args['results_root'], gen_ckpt_path)
    gen, opt_gen = load_model_optim(gen_ckpt_path, gen, opt_gen)
    if not os.path.exists(dis_ckpt_path):
        dis_ckpt_path = os.path.join(args['results_root'], dis_ckpt_path)
    dis, opt_dis = load_model_optim(dis_ckpt_path, dis, opt_dis)
    return Dict2Args(args), gen, opt_gen, dis, opt_dis


def load_model_optim(checkpoint_path, model=None, optim=None):
    """Load trained weight.

    Args:
        checkpoint_path (str)
        model (nn.Module)
        optim (torch.optim)

    Returns:
        model
        optim

    """

    ckpt = torch.load(checkpoint_path)
    if model is not None:
        model.load_state_dict(ckpt['model'])
        print("Load model ckpt from {}.".format(checkpoint_path))
    if optim is not None:
        optim.load_state_dict(ckpt['opt'])
        print("Load optim state from {}.".format(checkpoint_path))
    return model, optim


def load_model(checkpoint_path, model):
    """Load trained weight.

    Args:
        checkpoint_path (str)
        model (nn.Module)

    Returns:
        model

    """

    return load_model_optim(checkpoint_path, model, None)[0]


def load_optim(checkpoint_path, optim):
    """Load optimizer from checkpoint.

    Args:
        checkpoint_path (str)
        optim (torch.optim)

    Returns:
        optim

    """

    return load_model_optim(checkpoint_path, None, optim)[1]


def save_tensor_images(images, filename, nrow=None, normalize=True):
    if not nrow:
        torchvision.utils.save_image(images, filename, normalize=normalize, padding=0)
    else:
        torchvision.utils.save_image(images, filename, normalize=normalize, nrow=nrow, padding=0)


def prepare_results_dir(args):
    """Makedir, init tensorboard if required, save args."""
    root = os.path.join(args.results_root, args.data_name, args.target_model, "{}_{}".format(args.defense, args.timestamp))
    os.makedirs(root, exist_ok=True)
    if not args.no_tensorboard:
        from tensorboardX import SummaryWriter
        writer_root = os.path.join(root, "tensorboard_logs")
        os.makedirs(writer_root, exist_ok=True)
        writer = SummaryWriter(writer_root)
        print("-"*50, "Using tensorboard", "-"*50)
    else:
        writer = None

    train_image_root = os.path.join(root, "preview", "train")
    eval_image_root = os.path.join(root, "preview", "eval")
    os.makedirs(train_image_root, exist_ok=True)
    os.makedirs(eval_image_root, exist_ok=True)

    args.results_root = root
    args.train_image_root = train_image_root
    args.eval_image_root = eval_image_root

    if args.num_classes > args.n_eval_batches:
        args.n_eval_batches = args.num_classes
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size // 4

    if args.calc_FID:
        args.n_fid_batches = args.n_fid_images // args.batch_size

    args.device = "cuda"
    with open(os.path.join(root, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))
    return args, writer


def resume_from_dir(args, prev_timestamp):
    """

    Args:
        args:
        timestamp:

    Returns:

    """

    from models.generators import resnet64
    from models.discriminators import snresnet64
    from models.classifiers import VGG16, FaceNet, IR152, FaceNet64

    root = os.path.join(args.results_root, args.data_name, args.target_model, "{}_{}".format(args.defense, prev_timestamp))
    args_path = os.path.join(root, 'args.json')
    with open(args_path) as f:
        args = json.load(f)
    assert root == args['results_root']
    conditional = args['cGAN']
    num_classes = args['num_classes'] if conditional else 0
    # Initialize generator
    gen = resnet64.ResNetGenerator(
        args['gen_num_features'], args['gen_dim_z'], args['gen_bottom_width'],
        num_classes=num_classes, distribution=args['gen_distribution']
    )
    opt_gen = torch.optim.Adam(
        gen.parameters(), args['lr'], (args['beta1'], args['beta2'])
    )
    # Initialize discriminator
    args['dis_arch_concat'] = False
    if args['dis_arch_concat']:
        dis = snresnet64.SNResNetConcatDiscriminator(
            args['dis_num_features'], num_classes, dim_emb=args['dis_emb']
        )
    else:
        dis = snresnet64.SNResNetProjectionDiscriminator(
            args['dis_num_features'], num_classes
        )
    opt_dis = torch.optim.Adam(
        dis.parameters(), args['lr'], (args['beta1'], args['beta2'])
    )
    gen_ckpt_path, dis_ckpt_path = "gen_latest.pth.tar", "dis_latest.pth.tar"
    gen_ckpt_path = os.path.join(args['results_root'], gen_ckpt_path)
    dis_ckpt_path = os.path.join(args['results_root'], dis_ckpt_path)
    if os.path.exists(gen_ckpt_path):
        gen, opt_gen = load_model_optim(gen_ckpt_path, gen, opt_gen)
        print("load generator from {}".format(gen_ckpt_path))
    else:
        print("Warning: gen_ckpt not found: {}".format(gen_ckpt_path))
    if os.path.exists(dis_ckpt_path):
        dis, opt_dis = load_model_optim(dis_ckpt_path, dis, opt_dis)
        print("load discriminator from {}".format(dis_ckpt_path))
    else:
        print("Warning: dis_ckpt not found: {}".format(dis_ckpt_path))

    # Initialize classifier
    if 'lr_cls' in list(args.keys()):
        if args['target_model'] == ("VGG16"):
            cls = VGG16(num_classes)
            print("target_model = VGG16")
        elif args['target_model'] == ('IR152'):
            cls = IR152(num_classes)
            print("target_model = IR152")
        elif args['target_model'] == ("FaceNet64"):
            cls = FaceNet64(num_classes)
            print("target_model = FaceNet64")
        else:
            raise Exception("Invalid target_model name: {}".format(args['target_model']))
        cls = torch.nn.DataParallel(cls).cuda()
        opt_cls = torch.optim.SGD(cls.parameters(), args['lr_cls'], momentum=0.9, weight_decay=1e-4)
        cls_ckpt_path = "cls_latest.pth.tar"
        cls_ckpt_path = os.path.join(args['results_root'], cls_ckpt_path)
        if os.path.exists(cls_ckpt_path):
            cls, cls_dis = load_model_optim(cls_ckpt_path, cls, opt_cls)
            print("load classifier from {}".format(cls_ckpt_path))
        else:
            print("Warning: cls_ckpt not found: {}".format(cls_ckpt_path))
    else:
        cls, opt_cls = None, None

    return Dict2Args(args), gen, opt_gen, dis, opt_dis, cls, opt_cls


def load_target_model(args):
    # load target model
    print("Target Model:", args.target_model)
    if args.target_model == "VGG16":
        if args.defense == "bido":
            target_model = defense_models.VGG16(1000, True)
            target_model_path = 'checkpoints/target_model/VGG16_0.050&0.500_80.35.tar'
        else:
            target_model = classifiers.VGG16(args.num_classes)
            if args.data_name == "celeba":
                target_model_path = 'checkpoints/target_model/VGG16_88.26.tar'
            elif args.data_name == "vggface":
                target_model_path = 'PLG_MI_Results/vggface/VGG16/target_model/allclass_best.tar'
            elif args.data_name == "vggface2":
                target_model_path = 'PLG_MI_Results/vggface2/VGG16/target_model/allclass_best.tar'
            else:
                raise Exception("Invalid data name: {}".format(args.data_name))
    elif args.target_model == 'IR152':
        target_model = classifiers.IR152(args.num_classes)
        target_model_path = 'checkpoints/target_model/IR152_91.16.tar'
    elif args.target_model == "FaceNet64":
        target_model = classifiers.FaceNet64(args.num_classes)
        target_model_path = 'checkpoints/target_model/FaceNet64_88.50.tar'
    elif args.target_model == "FaceNet":
        target_model = classifiers.FaceNet(args.num_classes)
        target_model_path = 'checkpoints/evaluate_model/FaceNet_95.88.tar'
    else:
        raise Exception("Invalid target_model name: {}".format(args.target_model))

    return target_model, target_model_path
