from torchvision import transforms

image_transforms = transforms.Compose([
    transforms.Lambda(lambda img: _crop(img)),
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])
])

image_transforms_test = transforms.Compose([
    transforms.Lambda(lambda img: _crop(img)),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])
])

def _crop(img):
    r1 = 0.7
    r2 = 0.25
    r = 0.75
    w, h = img.size
    h_t = int(h * r1)
    w_t = int(r * w)
    x_t = int(w * r2)
    assert(h_t > 0  and h_t <= h)
    return img.crop((x_t, 0, w_t, h_t))

