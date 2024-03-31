import torch
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import numpy as np
from prepro import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def caption_image_beam_search(encoder, decoder, image, word_map, beam_size=3):
    k = beam_size
    vocab_size = len(word_map)

    # Encode
    encoder_out = encoder(image)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

    # Initialize variables
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1).to(device)
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Decode
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        awe, alpha = decoder.attention(encoder_out, h)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)
        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        prev_word_inds = (top_k_words / vocab_size).long()
        next_word_inds = (top_k_words % vocab_size).long()
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

        '''incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))'''
        
        '''if len(complete_inds) == 0:
            complete_seqs.extend(seqs.tolist())
            complete_seqs_alpha.extend(seqs_alpha.tolist())
            complete_seqs_scores.extend(top_k_scores.tolist())'''
        '''if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])'''
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        '''if len(complete_inds) == 0:
            complete_seqs.extend(seqs.tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])'''
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        
        k -= len(complete_inds)

        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > 50:
            break
        step += 1
    if len(complete_seqs_scores) == 0:
        return seqs[0], 0
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas

def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        # print(np.ceil(len(words) / 5.))
        plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()

def main(args):
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load trained model
    checkpoint = torch.load(args.checkpoint, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    image = Image.open(args.image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, image, vocab.word2idx, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention
    visualize_att(args.image_path, seq, alphas, vocab.idx2word, args.smooth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning with Beam Search')

    parser.add_argument('--checkpoint', type=str, default='result/model_pth/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar', help='path for trained model checkpoint')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_path', type=str, default='data/train2014/chicken_image_16.jpg', help='path to input image')
    parser.add_argument('--beam_size', type=int, default=3, help='beam size for beam search')
    parser.add_argument('--smooth', action='store_true', help='smooth attention overlay')

    args = parser.parse_args()
    main(args)
