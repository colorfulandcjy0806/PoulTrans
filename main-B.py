import time
from prepro import *
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from models.model_resnet50 import EncoderCNN, AttnDecoderRNN
from data_loader import get_loader
from nltk.translate.bleu_score import corpus_bleu
from utils import *
from rouge import Rouge
from Loss.ContrastiveMultiModalLoss import EnhancedLoss
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

seed = 1234
seed_everything(seed)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='B_model')
parser.add_argument('--model_path', type=str, default='', help='path for saving trained models')
parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
parser.add_argument('--image_dir', type=str, default='data/train2014', help='directory for resized images')
parser.add_argument('--image_dir_val', type=str, default='data/val2014', help='directory for resized images')
parser.add_argument('--caption_path', type=str, default='data/train_caption.json',
                    help='path for train annotation json file')
parser.add_argument('--caption_path_val', type=str, default='data/val_caption.json',
                    help='path for val annotation json file')
parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

# Model parameters
parser.add_argument('--embed_dim', type=int, default=1024, help='dimension of word embedding vectors')
parser.add_argument('--nhead', type=int, default=8, help='the number of heads in the multiheadattention models')
parser.add_argument('--num_layers', type=int, default=4,
                    help='the number of sub-encoder-layers in the transformer model')

parser.add_argument('--attention_dim', type=int, default=464, help='dimension of attention linear layers')
parser.add_argument('--decoder_dim', type=int, default=1024, help='dimension of decoder rnn')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--epochs_since_improvement', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--encoder_lr', type=float, default=1e-4)
parser.add_argument('--decoder_lr', type=float, default=4e-4)
parser.add_argument('--checkpoint', type=str, default=None, help='path for checkpoints')
parser.add_argument('--grad_clip', type=float, default=5.)
parser.add_argument('--alpha_c', type=float, default=1.)
parser.add_argument('--accumulate_best_bleu4', type=float, default=0.)
parser.add_argument('--fine_tune_encoder', type=bool, default='False', help='fine-tune encoder')

args = parser.parse_args()


# print(args)

def main(args):
    global best_accumulate_bleu4, epoch, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, vocab, bleu1, bleu2, bleu3, bleu4, rouge_1, rouge_2, rouge_l, cider,spice
    best_epoch = 0
    best_rouge = {'rouge-1': {'r': 0, 'p': 0, 'f': 0}, 'rouge-2': {'r': 0, 'p': 0, 'f': 0}, 'rouge-l': {'r': 0, 'p': 0, 'f': 0}}
    best_cider = 0.0
    best_spice = 0.0
    sm = 0.0
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # vocab = CustomUpickler(open(args.vocab_path, 'rb')).load()
    vocab_size = len(vocab)
    num_layers = 2
    d_model = 512
    num_heads = 8
    dff = 512
    dropout = 0.1
    if args.checkpoint is None:
        decoder = AttnDecoderRNN(attention_dim=1024, vocab_size=len(vocab),
                                 embed_dim=args.embed_dim,
                                 decoder_dim=1024,
                                 dropout=args.dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args.decoder_lr)
        encoder = EncoderCNN()
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr) if args.fine_tune_encoder else None


    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        bleu1 = checkpoint['bleu-1']
        bleu2 = checkpoint['bleu-2']
        bleu3 = checkpoint['bleu-3']
        bleu4 = checkpoint['bleu-4']
        accumulate_bleu4 = checkpoint['accumulate_bleu-4']
        rouge_1 = checkpoint['rouge["rouge-1"]']
        rouge_2 = checkpoint['rouge["rouge-2"]']
        rouge_l = checkpoint['rouge["rouge-l"]']
        cider = checkpoint['cider']
        spice = checkpoint['spice']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # criterion = ContrastiveTextImageLoss().to(device)
    criterion = EnhancedLoss(device = device).to(device)

    data_transforms = {
        # data_transforms是一个字典，包含了两种数据变换（'train' 和 'valid'），分别用于训练和验证数据集。
        'train':
            transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        'valid':
            transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        # transforms.ToTensor() 和 transforms.Normalize() 用于将图像数据转换为张量并进行标准化。
    }
    # Build data loader
    train_loader = get_loader(args.image_dir, args.caption_path, vocab,
                              data_transforms['train'], args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    val_loader = get_loader(args.image_dir_val, args.caption_path_val, vocab,
                            data_transforms['valid'], args.batch_size,
                            shuffle=True, num_workers=args.num_workers)

    best_accumulate_bleu4 = 0.
    best_bleu1 = 0.
    best_bleu2 = 0.
    best_bleu3 = 0.
    best_bleu4 = 0.
    best_cider = 0.
    best_rouge = {'rouge-1': {'r': 0, 'p': 0, 'f': 0}, 'rouge-2': {'r': 0, 'p': 0, 'f': 0}, 'rouge-l': {'r': 0, 'p': 0, 'f': 0}}
    best_spice = 0.
    

    for epoch in range(args.start_epoch, args.epochs):
        '''if args.epochs_since_improvement == 20:
            break'''  
        if args.epochs_since_improvement > 0 and args.epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if args.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        recent_accumulate_bleu4, recent_bleu1, recent_bleu2, recent_bleu3, recent_bleu4, recent_cider, recent_rouge, recent_spice = validate(
            val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion)



        is_best = recent_accumulate_bleu4 > best_accumulate_bleu4
        best_accumulate_bleu4 = max(recent_accumulate_bleu4, best_accumulate_bleu4)
        best_bleu1 = max(recent_bleu1, best_bleu1)
        best_bleu2 = max(recent_bleu2, best_bleu2)
        best_bleu3 = max(recent_bleu3, best_bleu3)
        best_bleu4 = max(recent_bleu4, best_bleu4)
        best_rouge['rouge-1']['f'] = max(recent_rouge['rouge-1']['f'], best_rouge['rouge-1']['f'])
        best_rouge['rouge-2']['f'] = max(recent_rouge['rouge-2']['f'], best_rouge['rouge-2']['f'])
        best_rouge['rouge-l']['f'] = max(recent_rouge['rouge-l']['f'], best_rouge['rouge-l']['f'])
        best_accumulate_bleu4 = max(recent_accumulate_bleu4,best_accumulate_bleu4)
        best_cider = max(recent_cider, best_cider)
        best_spice = max(recent_spice, best_spice)


        if recent_bleu1 > best_bleu1:
            best_bleu1 = recent_bleu1
        print(f'Best BLEU-1: {best_bleu1}')

        if recent_bleu2 > best_bleu2:
            best_bleu2 = recent_bleu2
        print(f'Best BLEU-2: {best_bleu2}')

        if recent_bleu3 > best_bleu3:
            best_bleu3 = recent_bleu3
        print(f'Best BLEU-3: {best_bleu3}')

        if recent_bleu4 > best_bleu4:
            best_bleu4 = recent_bleu4
        print(f'Best BLEU-4: {best_bleu4}')

        # 更新最佳 Rouge 和 CIDEr 指标
        if recent_rouge['rouge-1']['f'] > best_rouge['rouge-1']['f']:
            best_rouge['rouge-1'] = recent_rouge['rouge-1']
        print(f'Best rouge_1: {best_rouge["rouge-1"]}')
        if recent_rouge['rouge-2']['f'] > best_rouge['rouge-2']['f']:
            best_rouge['rouge-2'] = recent_rouge['rouge-2']
        print(f'Best rouge_2: {best_rouge["rouge-2"]}')
        if recent_rouge['rouge-l']['f'] > best_rouge['rouge-l']['f']:
            best_rouge['rouge-l'] = recent_rouge['rouge-l']
        print(f'Best rouge_l: {best_rouge["rouge-l"]}')
        if recent_cider > best_cider:
            best_cider = recent_cider
        print(f'Best CIDEr: {best_cider}')
        if recent_accumulate_bleu4 > best_accumulate_bleu4:
            best_accumulate_bleu4 = recent_accumulate_bleu4
        print(f'Best accumulate bleu4: {best_accumulate_bleu4}')
        if recent_spice > best_spice:
            best_spice = best_spice
        print(f'Best spice score: {best_spice}')
        #计算总指标Sm
        Sm = (best_bleu4 + best_cider + best_rouge["rouge-l"]['f'] + best_spice) / 4
        print(f'Sm: {Sm}')

        if not is_best:
            args.epochs_since_improvement += 1
            print("\nEpoch since last improvement: %d\n" % (args.epochs_since_improvement,))
        else:
            args.epochs_since_improvement = 0

    save_checkpoint(args.data_name, epoch, args.epochs_since_improvement, encoder, decoder, encoder_optimizer,
                decoder_optimizer, best_accumulate_bleu4, is_best, best_bleu1, best_bleu2, best_bleu3,
                best_bleu4, best_rouge, best_cider, best_spice)



def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1accs = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    # Use tqdm to display progress bar
    with tqdm(total=len(train_loader)) as t:
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            data_time.update(time.time() - start)

            # Move to GPU, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, _ = decoder(imgs, caps, caplens)
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = caps[:, 1:]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            scores = scores.data.to(device)
            targets = targets.data.to(device)
            #print("imgs: ", imgs.shape)
            #print("scores: ", scores.shape)
            #print("targets: ", targets.shape)
            loss = criterion(imgs, scores, targets).to(device)
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            if args.grad_clip is not None:
                clip_gradient(decoder_optimizer, args.grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, args.grad_clip)

            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            top1 = accuracy(scores, targets, 1)
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top1accs.update(top1, sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Update tqdm progress bar
            t.set_postfix(loss=losses.avg, top1=top1accs.avg, top5=top5accs.avg)
            t.update()

    # Print status
    print('Epoch: [{0}]\t'
          'Loss {loss.avg:.4f}\t'
          'Top-1 Accuracy {top1.avg:.3f}\t'
          'Top-5 Accuracy {top5.avg:.3f}'.format(epoch, loss=losses, top1=top1accs, top5=top5accs))

    # Return average loss and accuracy
    return losses.avg, top1accs.avg, top5accs.avg


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: accumulate-BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1accs = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating accumulate-BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    for i, (imgs, caps, caplens) in enumerate(val_loader):

        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)

        # Forward prop.
        if encoder is not None:
            imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, _ = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = caps[:, 1:]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        scores = scores.data.to(device)
        targets = targets.data.to(device)
        # print(scores)
        # Calculate loss
        loss = criterion(imgs ,scores, targets).to(device)

        # Add doubly stochastic attention regularization
        # loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean().to(device)

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        top1 = accuracy(scores, targets, 1)
        top5 = accuracy(scores, targets, 5)
        top1accs.update(top1, sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % args.log_step == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                            loss=losses, top1=top1accs, top5=top5accs))

        # Store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
# 清空references和hypotheses列表
        references.clear()
        hypotheses.clear()

        # References
        for j in range(caps_sorted.shape[0]):
            img_caps = caps_sorted[j].tolist()
            img_captions = [w for w in img_caps if w not in {vocab.__call__('<start>'), vocab.__call__('<pad>')}]
            references.append([img_captions])

        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = [preds[j][:decode_lengths[j]] for j in range(len(preds))]  # 使用列表推导来简化逻辑
        hypotheses.extend(temp_preds)

        assert len(references) == len(hypotheses)



    # Calculate accumulate-BLEU-4 scores
    accumulate_bleu4 = corpus_bleu(references, hypotheses)
    # print("references: ", references)
    # print("hypotheses: ", hypotheses)
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))

    # 计算 ROUGE

    # References
    references_text = []  # 存储参考字幕的文本
    for ref in references:
        ref_text = " ".join([" ".join([vocab.idx2word[word_idx] for word_idx in cap]) for cap in ref])
        references_text.append(ref_text)

    # Hypotheses
    hypotheses_text = []  # 存储假设字幕的文本
    for hyp in hypotheses:
        hyp_text = " ".join([vocab.idx2word[word_idx] for word_idx in hyp])
        hypotheses_text.append(hyp_text)

    
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypotheses_text, references_text, avg=True)


    # 将参考文本转换为字符串格式
    refs_str = []
    for ref_list in references:
        ref_str = [' '.join([vocab.idx2word[word_idx] for word_idx in cap]) for cap in ref_list]
        refs_str.append(ref_str)

    # 将假设文本转换为字符串格式
    hypotheses_str = [' '.join([vocab.idx2word[word_idx] for word_idx in hyp]) for hyp in hypotheses]

    # 将数据转换为适用于CIDEr计算的格式
    gts = {idx: ref_list for idx, ref_list in enumerate(refs_str)}
    res = {idx: [hyp] for idx, hyp in enumerate(hypotheses_str)}

    # 计算CIDEr分数
    cider = Cider()
    cider_score, _ = cider.compute_score(gts=gts, res=res)

# Convert references to SPICE format
    '''refs_spice_format = {}
    for idx, ref_list in enumerate(refs_str):
        refs_spice_format[idx] = ref_list

    # Convert hypotheses to SPICE format
    res_spice_format = {idx: [hyp] for idx, hyp in enumerate(hypotheses_str)}

    # Calculate SPICE scores'''
    spice = Spice()
    #print("gts:", gts)
    #print("res:", res)
    spice_score, _ = spice.compute_score(gts, res)
                                        #gts=refs_spice_format, res=res_spice_format





    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-1 ACCURACY - {top1.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, accumulate_BLEU-4  {accumulate_bleu4}, bleu-1 {bleu1}, bleu-2 {bleu2}, bleu-3 {bleu3}, bleu-4 {bleu4}, CIDEr score {cider_score}, ROUGE scores {rouge_scores},SPICE scores {spice_score}\n'.format(
            loss=losses,
            top1=top1accs,
            top5=top5accs,
            accumulate_bleu4=accumulate_bleu4,
            bleu1=bleu1,
            bleu2=bleu2,
            bleu3=bleu3,
            bleu4=bleu4,
            cider_score=cider_score,
            rouge_scores=rouge_scores,
            spice_score=spice_score))
    return accumulate_bleu4, bleu1, bleu2, bleu3, bleu4, cider_score, rouge_scores,spice_score



if __name__ == '__main__':
    main(args)
