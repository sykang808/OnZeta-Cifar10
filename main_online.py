import argparse
import torch
import torch.nn.functional as F
import clip
import math
import torchvision.transforms as transforms
import torchvision

# 사용 가능한 CLIP 모델 리스트
model_names = ['RN50', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

# 커맨드 라인 인자 파서 설정
parser = argparse.ArgumentParser(description='OnZeta for Image Classification')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'], help='dataset to use')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: RN50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--tau_t', default=0.01, type=float, help='text temperature')
parser.add_argument('--tau_i', default=0.04, type=float, help='image temperature')
parser.add_argument('--cw', default=0.5, type=float, help='vision proxy learning rate')
parser.add_argument('--cr', default=20, type=float, help='dual variable learning rate')
parser.add_argument('--alpha', default=1, type=float, help='class balance parameter')
parser.add_argument('--beta', default=0.8, type=float, help='text-vision mixing ratio')
parser.add_argument('--repeat', default=5, type=int, help='number of repetitions')

def main():
    args = parser.parse_args()
    print(args)

    # CLIP 모델 로드
    print('Loading pre-trained model...')
    model, preprocess = clip.load(args.arch)
    model = model.cuda()
    model.eval()

    # 데이터셋 설정
    if args.dataset == 'cifar10':
        num_classes = 10
        dataset_class = torchvision.datasets.CIFAR10
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        templates = [
            'a photo of a {}.',
            'a colored photo of a {}.',
            'a photo of the {}.',
            'a close-up photo of a {}.',
            'a bright photo of a {}.',
            'a cropped photo of a {}.',
            'a photo of a clean {}.',
        ]
    elif args.dataset == 'cifar100':
        num_classes = 100
        dataset_class = torchvision.datasets.CIFAR100
        classes = [f'class_{i}' for i in range(100)]  # CIFAR100 클래스 이름 설정 필요
        templates = [
            'a photo of a {}.',
            'a photo of the {}.',
        ]

    # 데이터 로드 및 전처리
    print('Loading data...')
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    val_set = dataset_class(root='./data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # 이미지 특징 추출
    print('Extracting image features...')
    image_feat, image_label = extract_features(model, loader)
    n = len(image_label)

    # 텍스트 프록시 얻기
    print('Obtaining text proxy...')
    text_classifier = zeroshot_classifier(model, classes, templates)
    text_classifier = text_classifier.float()
    logits_t = image_feat @ text_classifier
    acc1, acc5 = accuracy(logits_t, image_label, topk=(1, 5))
    print(f'Accuracy with text proxy: Top-1 {acc1:.2f}%, Top-5 {acc5:.2f}%')


    # OnZeta 알고리즘 실행
    print(f'Running online zero-shot transfer: repeat {args.repeat} times')
    acc_onzeta = torch.zeros(args.repeat).cuda()
    acc_onlab = torch.zeros(args.repeat).cuda()

    for iter in range(args.repeat):
        acc_onlab[iter], acc_onzeta[iter] = run_onzeta(args, image_feat, image_label, text_classifier, n, num_classes)
        print(f'Iteration {iter+1}: OnLab Acc: {acc_onlab[iter]:.2f}%, OnZeta Acc: {acc_onzeta[iter]:.2f}%')

    # 최종 결과 출력
    print(f'Dataset: {args.dataset}')
    print(f'Mean accuracy of OnLab: {torch.mean(acc_onlab):.2f}% ± {torch.std(acc_onlab):.2f}%')
    print(f'Mean accuracy of OnZeta: {torch.mean(acc_onzeta):.2f}% ± {torch.std(acc_onzeta):.2f}%')


def extract_features(model, loader):
    image_feat = []
    image_label = []
    with torch.no_grad():
        for images, target in loader:
            images = images.cuda()
            target = target.cuda()
            image_features = model.encode_image(images)
            image_feat.append(F.normalize(image_features, dim=1))
            image_label.append(target)
    return torch.cat(image_feat, dim=0).float(), torch.cat(image_label, dim=0)

def run_onzeta(args, image_feat, image_label, text_classifier, n, num_classes):
    idx = torch.randperm(n).cuda()
    combo_label = torch.zeros(n, num_classes).cuda()
    text_label = torch.zeros(n, num_classes).cuda()
    w = text_classifier.clone()
    rho = torch.zeros(num_classes).cuda()

    for i in range(n):
        lr = args.cw / math.sqrt(i + 1)
        rlr = args.cr / math.sqrt(i + 1)
        beta = args.beta * math.sqrt((i + 1) / n)
        x = image_feat[idx[i], :]
        
        tlabel = F.softmax(x @ text_classifier / args.tau_t, dim=0)
        tlabel = tlabel * torch.exp(rho)
        tlabel /= torch.sum(tlabel)
        rho -= rlr * (tlabel - args.alpha / num_classes)
        rho[rho < 0] = 0
        text_label[i, :] = tlabel
        
        vision_label = F.softmax(x @ w / args.tau_i, dim=0)
        combo_label[i, :] = beta * vision_label + (1 - beta) * tlabel
        
        grad = torch.outer(x, vision_label - tlabel)
        w -= (lr / args.tau_i) * grad
        w = F.normalize(w, dim=0)

    acc_onlab = accuracy(text_label, image_label[idx], topk=(1,))
    acc_onzeta = accuracy(combo_label, image_label[idx], topk=(1,))
    return acc_onlab, acc_onzeta
    
def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res[0] if len(res) == 1 else res


if __name__ == '__main__':
    main()
