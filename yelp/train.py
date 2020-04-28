import argparse
import os
import time
import math
import numpy as np
import random
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import to_gpu, Corpus, batchify
from models import Seq2Seq2Decoder, Seq2Seq, MLP_D, MLP_G, MLP_Classify, generate, saving_models
import shutil

parser = argparse.ArgumentParser(description='ARAE for Yelp transfer')
# Path Arguments
parser.add_argument('--data_path', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--outf', type=str, default='yelp_example',
                    help='output directory name')
parser.add_argument('--load_vocab', type=str, default="",
                    help='path to load vocabulary from')

parser.add_argument('--saving_model', type=str, default=True,
                    help='save the models')

parser.add_argument('--model_saving_path', type=str, default="./saved_models",
                    help='path to save models')

# Data Processing Arguments
parser.add_argument('--dataset_name', type=str, default='twitter',
                    help='the name of the dataset used')

parser.add_argument('--vocab_size', type=int, default=30000,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
parser.add_argument('--maxlen', type=int, default=25,
                    help='maximum sentence length')
parser.add_argument('--lowercase', dest='lowercase', action='store_true',
                    help='lowercase all text')
parser.add_argument('--no-lowercase', dest='lowercase', action='store_true',
                    help='not lowercase all text')
parser.set_defaults(lowercase=True)

# Model Arguments
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhidden', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--noise_r', type=float, default=0.1,
                    help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--noise_anneal', type=float, default=0.9995,
                    help='anneal noise_r exponentially by this'
                         'every 100 iterations')
parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
parser.add_argument('--arch_g', type=str, default='128-128',
                    help='generator architecture (MLP)')
parser.add_argument('--arch_d', type=str, default='128-128',
                    help='critic/discriminator architecture (MLP)')
parser.add_argument('--arch_classify', type=str, default='128-128',
                    help='classifier architecture')
parser.add_argument('--z_size', type=int, default=32,
                    help='dimension of random noise z to feed into generator')
parser.add_argument('--temp', type=float, default=1,
                    help='softmax temperature (lower --> more discrete)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')

# Training Arguments
parser.add_argument('--epochs', type=int, default=50,
                    help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--niters_ae', type=int, default=1,
                    help='number of autoencoder iterations in training')
parser.add_argument('--niters_gan_d', type=int, default=5,
                    help='number of discriminator iterations in training')
parser.add_argument('--niters_gan_g', type=int, default=1,
                    help='number of generator iterations in training')
parser.add_argument('--niters_gan_ae', type=int, default=1,
                    help='number of gan-into-ae iterations in training')
parser.add_argument('--niters_gan_schedule', type=str, default='',
                    help='epoch counts to increase number of GAN training '
                         ' iterations (increment by 1 each time)')
parser.add_argument('--lr_ae', type=float, default=1,
                    help='autoencoder learning rate')
parser.add_argument('--lr_gan_g', type=float, default=1e-04,
                    help='generator learning rate')
parser.add_argument('--lr_gan_d', type=float, default=1e-04,
                    help='critic/discriminator learning rate')
parser.add_argument('--lr_classify', type=float, default=1e-04,
                    help='classifier learning rate')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')
parser.add_argument('--gan_gp_lambda', type=float, default=0.1,
                    help='WGAN GP penalty lambda')
parser.add_argument('--grad_lambda', type=float, default=0.01,
                    help='WGAN into AE lambda')
parser.add_argument('--lambda_class', type=float, default=1,
                    help='lambda on classifier')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--log_interval', type=int, default=200,
                    help='interval to log autoencoder training results')

# Other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', dest='cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--no-cuda', dest='cuda', action='store_true',
                    help='not using CUDA')
parser.set_defaults(cuda=True)
parser.add_argument('--device_id', type=str, default='0')

args = parser.parse_args()
print(vars(args))

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

# make output directory if it doesn't already exist
if os.path.isdir(args.outf):
    shutil.rmtree(args.outf)
os.makedirs(args.outf)

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

label_ids = {"pos": 1, "neg": 0}
id2label = {1:"pos", 0:"neg"}

# (Path to textfile, Name, Use4Vocab)
datafiles = [(os.path.join(args.data_path, "valid_{}.txt".format(args.dataset_name)), "valid", False),
             (os.path.join(args.data_path, "train_{}.txt").format(args.dataset_name), "train", True),
             (os.path.join(args.data_path, "test_{}.txt").format(args.dataset_name), "test", True)]
vocabdict = None
if args.load_vocab != "":
    vocabdict = json.load(args.vocab)
    vocabdict = {k: int(v) for k, v in vocabdict.items()}
corpus = Corpus(datafiles,
                maxlen=args.maxlen,
                vocab_size=args.vocab_size,
                lowercase=args.lowercase,
                vocab=vocabdict)

# dumping vocabulary
with open('{}/{}_vocab.json'.format(args.outf, args.dataset_name), 'w') as f:
    json.dump(corpus.dictionary.word2idx, f)

# save arguments
ntokens = len(corpus.dictionary.word2idx)
print("Vocabulary Size: {}".format(ntokens))
args.ntokens = ntokens
with open('{}/{}_args.json'.format(args.outf, args.dataset_name), 'w') as f:
    json.dump(vars(args), f)
with open("{}/{}_log.txt".format(args.outf, args.dataset_name), 'w') as f:
    f.write(str(vars(args)))
    f.write("\n\n")

eval_batch_size = 100
valid_data = batchify(corpus.data['valid'], eval_batch_size, shuffle=False)
train_data = batchify(corpus.data['train'], args.batch_size, shuffle=True)
test_data = batchify(corpus.data['test'], args.batch_size, shuffle=True)
print("Loaded data!")

###############################################################################
# Build the models
###############################################################################

ntokens = len(corpus.dictionary.word2idx)
autoencoder = Seq2Seq(emsize=args.emsize,
                      nhidden=args.nhidden,
                      ntokens=ntokens,
                      nlayers=args.nlayers,
                      noise_r=args.noise_r,
                      hidden_init=args.hidden_init,
                      dropout=args.dropout,
                      gpu=args.cuda)

gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
gan_disc = MLP_D(ninput=args.nhidden, noutput=1, layers=args.arch_d)
g_factor = None

print(autoencoder)
print(gan_gen)
print(gan_disc)

optimizer_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
optimizer_gan_g = optim.Adam(gan_gen.parameters(),
                             lr=args.lr_gan_g,
                             betas=(args.beta1, 0.999))
optimizer_gan_d = optim.Adam(gan_disc.parameters(),
                             lr=args.lr_gan_d,
                             betas=(args.beta1, 0.999))

criterion_ce = nn.CrossEntropyLoss()

if args.cuda:
    autoencoder = autoencoder.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()
    # classifier = classifier.cuda()
    criterion_ce = criterion_ce.cuda()

###############################################################################
# Training code
###############################################################################


def save_model():
    print("Saving models")
    with open('{}/autoencoder_model.pt'.format(args.outf), 'wb') as f:
        torch.save(autoencoder.state_dict(), f)
    with open('{}/gan_gen_model.pt'.format(args.outf), 'wb') as f:
        torch.save(gan_gen.state_dict(), f)
    with open('{}/gan_disc_model.pt'.format(args.outf), 'wb') as f:
        torch.save(gan_disc.state_dict(), f)

def grad_hook_cla(grad):
    return grad * args.lambda_class


def evaluate_autoencoder(data_source, epoch):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary.word2idx)
    all_accuracies = 0
    bcnt = 0
    for i, batch in enumerate(data_source):
        source, target, lengths = batch
        source = to_gpu(args.cuda, Variable(source, volatile=True))
        target = to_gpu(args.cuda, Variable(target, volatile=True))

        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

        hidden = autoencoder(source, lengths, noise=False, encode_only=True)

        output = autoencoder(source, lengths, noise=False)
        flattened_output = output.view(-1, ntokens)
        masked_output = \
            flattened_output.masked_select(output_mask).view(-1, ntokens)
        # accuracy
        max_vals, max_indices = torch.max(masked_output, 1)
        all_accuracies += \
            torch.mean(max_indices.eq(masked_target).float()).data[0]

        max_vals, max_indices = torch.max(output, 2)
        total_loss += criterion_ce(masked_output/args.temp, masked_target).data
        bcnt += 1

        aeoutf_from = "{}/{}_{}_output_decoder_from.txt".format(args.outf, epoch, args.dataset_name)
        aeoutf_tran = "{}/{}_{}_output_decoder_tran.txt".format(args.outf, epoch, args.dataset_name)
        with open(aeoutf_from, 'w') as f_from, open(aeoutf_tran,'w') as f_trans:
            max_indices = \
                max_indices.view(output.size(0), -1).data.cpu().numpy()
            target = target.view(output.size(0), -1).data.cpu().numpy()
            tran_indices = max_indices
            for t, tran_idx in zip(target, tran_indices):
                # real sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in t])
                f_from.write(chars)
                f_from.write("\n")
                # transfer sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in tran_idx])
                f_trans.write(chars)
                f_trans.write("\n")

    return total_loss[0] / len(data_source), all_accuracies/bcnt


def evaluate_generator(noise, epoch):
    gan_gen.eval()
    autoencoder.eval()

    # generate from fixed random noise
    fake_hidden = gan_gen(noise)
    max_indices = \
        autoencoder.generate(fake_hidden, maxlen=50, sample=args.sample)

    with open("%s/%s_%s_generated.txt" % (args.outf, epoch, args.dataset_name), "w") as f:
        max_indices = max_indices.data.cpu().numpy()
        for idx in max_indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars)
            f.write("\n")


def train_ae(batch, total_loss_ae, start_time, i):
    autoencoder.train()
    optimizer_ae.zero_grad()

    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))

    mask = target.gt(0)
    masked_target = target.masked_select(mask)
    output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)
    output = autoencoder(source, lengths, noise=True)
    flat_output = output.view(-1, ntokens)
    masked_output = flat_output.masked_select(output_mask).view(-1, ntokens)
    loss = criterion_ce(masked_output/args.temp, masked_target)
    loss.backward()

    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)
    optimizer_ae.step()

    total_loss_ae += loss.data

    accuracy = None
    if i % args.log_interval == 0 and i > 0:
        probs = F.softmax(masked_output, dim=-1)
        max_vals, max_indices = torch.max(probs, 1)
        accuracy = torch.mean(max_indices.eq(masked_target).float()).data[0]
        cur_loss = total_loss_ae[0] / args.log_interval
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
              'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'
              .format(epoch, i, len(train_data),
                      elapsed * 1000 / args.log_interval,
                      cur_loss, math.exp(cur_loss), accuracy))

        with open("{}/{}_log.txt".format(args.outf, args.dataset_name), 'a') as f:
            f.write('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}\n'.
                    format(epoch, i, len(train_data),
                           elapsed * 1000 / args.log_interval,
                           cur_loss, math.exp(cur_loss), accuracy))

        total_loss_ae = 0
        start_time = time.time()

    return total_loss_ae, start_time


def train_gan_g():
    gan_gen.train()
    gan_gen.zero_grad()

    noise = to_gpu(args.cuda,
                   Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)
    fake_hidden = gan_gen(noise)
    errG = gan_disc(fake_hidden)
    errG.backward(one)
    optimizer_gan_g.step()

    return errG


def grad_hook(grad):
    return grad * args.grad_lambda


''' Steal from https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py '''
def calc_gradient_penalty(netD, real_data, fake_data):
    bsz = real_data.size(0)
    alpha = torch.rand(bsz, 1)
    alpha = alpha.expand(bsz, real_data.size(1))  # only works for 2D XXX
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gan_gp_lambda
    return gradient_penalty


def train_gan_d(batch):
    gan_disc.train()
    optimizer_gan_d.zero_grad()

    # positive samples ----------------------------
    # generate real codes
    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))

    # batch_size x nhidden
    real_hidden = autoencoder(source, lengths, noise=False, encode_only=True)

    # loss / backprop
    errD_real = gan_disc(real_hidden)
    errD_real.backward(one)

    # negative samples ----------------------------
    # generate fake codes
    noise = to_gpu(args.cuda,
                   Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)

    # loss / backprop
    fake_hidden = gan_gen(noise)
    errD_fake = gan_disc(fake_hidden.detach())
    errD_fake.backward(mone)

    # gradient penalty
    gradient_penalty = calc_gradient_penalty(gan_disc, real_hidden.data, fake_hidden.data)
    gradient_penalty.backward()

    optimizer_gan_d.step()
    errD = -(errD_real - errD_fake)

    return errD, errD_real, errD_fake


def train_gan_d_into_ae(batch):
    autoencoder.train()
    optimizer_ae.zero_grad()

    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))
    real_hidden = autoencoder(source, lengths, noise=False, encode_only=True)
    real_hidden.register_hook(grad_hook)
    errD_real = gan_disc(real_hidden)
    errD_real.backward(mone)
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)

    optimizer_ae.step()

    return errD_real


print("Training...")
with open("{}/{}_log.txt".format(args.outf, args.dataset_name), 'a') as f:
    f.write('Training...\n')

# schedule of increasing GAN training loops
if args.niters_gan_schedule != "":
    gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
else:
    gan_schedule = []
niter_gan = 1

fixed_noise = to_gpu(args.cuda,
                     Variable(torch.ones(args.batch_size, args.z_size)))
fixed_noise.data.normal_(0, 1)
one = to_gpu(args.cuda, torch.FloatTensor([1]))
mone = one * -1

for epoch in range(1, args.epochs+1):
    # update gan training schedule
    if epoch in gan_schedule:
        niter_gan += 1
        print("GAN training loop schedule increased to {}".format(niter_gan))
        with open("{}/{}_log.txt".format(args.outf, args.dataset_name), 'a') as f:
            f.write("GAN training loop schedule increased to {}\n".
                    format(niter_gan))

    total_loss_ae = 0
    total_loss_ae2 = 0
    classify_loss = 0
    epoch_start_time = time.time()
    start_time = time.time()
    niter = 0
    niter_global = 1

    # loop through all batches in training data
    while niter < len(train_data):

        # train autoencoder ----------------------------
        for i in range(args.niters_ae):
            if niter == len(train_data):
                break  # end of epoch
            total_loss_ae, start_time = \
                train_ae(train_data[niter], total_loss_ae, start_time, niter)

            niter += 1

        # train gan ----------------------------------
        for k in range(niter_gan):

            # train discriminator/critic
            for i in range(args.niters_gan_d):
                # feed a seen sample within this epoch; good for early training
                # if i % 2 == 0:
                batch = train_data[random.randint(0, len(train_data) - 1)]
                errD, errD_real, errD_fake = train_gan_d(batch)

            # train generator
            for i in range(args.niters_gan_g):
                errG = train_gan_g()

            # train autoencoder from d
            for i in range(args.niters_gan_ae):
                # if i % 2 == 0:
                batch = train_data[random.randint(0, len(train_data) - 1)]
                errD_ = train_gan_d_into_ae(batch)

        niter_global += 1
        if niter_global % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f (Loss_D_real: %.4f '
                  'Loss_D_fake: %.4f) Loss_G: %.4f'
                  % (epoch, args.epochs, niter, len(train_data),
                     errD.data[0], errD_real.data[0],
                     errD_fake.data[0], errG.data[0]))
            # print("Classify loss: {:5.2f} | Classify accuracy: {:3.3f}\n".format(
            #         classify_loss, classify_acc))
            with open("{}/{}_log.txt".format(args.outf, args.dataset_name), 'a') as f:
                f.write('[%d/%d][%d/%d] Loss_D: %.4f (Loss_D_real: %.4f '
                        'Loss_D_fake: %.4f) Loss_G: %.4f\n'
                        % (epoch, args.epochs, niter, len(train_data),
                           errD.data[0], errD_real.data[0],
                           errD_fake.data[0], errG.data[0]))
                # f.write("Classify loss: {:5.2f} | Classify accuracy: {:3.3f}\n".format(
                #         classify_loss, classify_acc))

            # exponentially decaying noise on autoencoder
            autoencoder.noise_r = \
                autoencoder.noise_r*args.noise_anneal

    # end of epoch ----------------------------
    # evaluation
    test_loss, accuracy = evaluate_autoencoder(valid_data[:1000], epoch)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
          'test ppl {:5.2f} | acc {:3.3f}'.
          format(epoch, (time.time() - epoch_start_time),
                 test_loss, math.exp(test_loss), accuracy))
    print('-' * 89)
    with open("{}/{}_log.txt".format(args.outf, args.dataset_name), 'a') as f:
        f.write('-' * 89)
        f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
                ' test ppl {:5.2f} | acc {:3.3f}\n'.
                format(epoch, (time.time() - epoch_start_time),
                       test_loss, math.exp(test_loss), accuracy))
        f.write('-' * 89)
        f.write('\n')

    evaluate_generator(fixed_noise, "end_of_epoch_{}".format(epoch))
    # shuffle between epochs
    train_data = batchify(corpus.data['train'], args.batch_size, shuffle=True)

saving_models(autoencoder, gan_gen, gan_disc, args.model_saving_path, args.dataset_name)

    
test_loss, accuracy = evaluate_autoencoder(valid_data, epoch + 1)
print('-' * 89)
print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
      'test ppl {:5.2f} | acc {:3.3f}'.
      format(epoch, (time.time() - epoch_start_time),
             test_loss, math.exp(test_loss), accuracy))
print('-' * 89)
with open("{}/{}_log.txt".format(args.outf, args.dataset_name), 'a') as f:
    f.write('-' * 89)
    f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
            ' test ppl {:5.2f} | acc {:3.3f}\n'.
            format(epoch, (time.time() - epoch_start_time),
                   test_loss, math.exp(test_loss), accuracy))
    f.write('-' * 89)
    f.write('\n')

if args.saving_model:
    saving_models(autoencoder, gan_gen, gan_disc, args.model_saving_path, args.dataset_name)

# data generation

index = np.random.randint(0,test_data.__len__())

source, target, lengths = test_data[index]
source = to_gpu(args.cuda, Variable(source))
target = to_gpu(args.cuda, Variable(target))
latent_representation_original_data = autoencoder(source, lengths, noise=False, encode_only=True)
generated_data_hidden = []
generated_sentences = []
for j in range(args.batch_size):
    noise = to_gpu(args.cuda,
                   Variable(torch.ones(1000, args.z_size)))
    noise.data.normal_(0, 1)
    fake_hidden = gan_gen(noise)
    generated_data_hidden.append(fake_hidden)
    max_indices = autoencoder.generate(hidden=fake_hidden,
                                       maxlen=args.maxlen,
                                       sample=args.sample)

    max_indices = max_indices.data.cpu().numpy()
    sentences = []
    for idx in max_indices:
        # generated sentence
        words = [corpus.dictionary.idx2word[x] for x in idx]
        # truncate sentences to first occurrence of <eos>
        truncated_sent = []
        for w in words:
            if w != '<eos>':
                truncated_sent.append(w)
            else:
                break
        sent = " ".join(truncated_sent)
        sentences.append(sent)
    generated_sentences.append(sentences)

with open('./yelp_example/latent_representation_original_data.txt', 'w') as f:
    for item in latent_representation_original_data:
        f.write("%s\n" % item)

with open('./yelp_example/generated_data_hidden.txt', 'w') as f:
    for item in generated_data_hidden:
        f.write("%s\n" % item)

with open('./yelp_example/generated_sentences.txt', 'w') as f:
    for item in generated_sentences:
        f.write("%s\n" % item)
