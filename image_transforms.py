from torchvision import transforms

image_transforms = transforms.Compose([
    transforms.Lambda(lambda img: _crop(img, 0.7)),
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])
])

def _crop(img, r):
    w, h = img.size
    h_t = int(h * r)
    assert(h_t > 0  and h_t <= h)
    return img.crop((0, 0, w, h_t))

