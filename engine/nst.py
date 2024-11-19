import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import time

# Load and preprocess images
def load_image(img_path, size=512):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Perform Neural Style Transfer
def neural_style_transfer(content_path, style_path, output_path, iterations=300, style_weight=1e6, content_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_image = load_image(content_path).to(device)
    style_image = load_image(style_path).to(device)

    model = models.vgg19(pretrained=True).features.to(device).eval()
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_features(image, model):
        layers = {
            '0': 'conv_1', '5': 'conv_2', '10': 'conv_3',
            '19': 'conv_4', '28': 'conv_5'
        }
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def gram_matrix(tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        return torch.mm(tensor, tensor.t())

    style_features = get_features(style_image, model)
    content_features = get_features(content_image, model)
    target = content_image.clone().requires_grad_(True).to(device)

    optimizer = torch.optim.Adam([target], lr=0.003)
    start_time = time.time()  # 시작 시간 기록
    for i in range(iterations):
        target_features = get_features(target, model)
        content_loss = F.mse_loss(target_features['conv_4'], content_features['conv_4'])

        style_loss = 0
        for layer in style_layers:
            target_gram = gram_matrix(target_features[layer])
            style_gram = gram_matrix(style_features[layer])
            layer_style_loss = F.mse_loss(target_gram, style_gram)
            style_loss += layer_style_loss / (target_gram.size()[0] ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        print(f'\r진행률: {(i+1)/iterations*100:.2f}%', end='')
    end_time = time.time()    # 종료 시간 기록
    execution_time = end_time - start_time
    print(f"실행 시간: {execution_time:.4f}초")

    # Save output
    output_image = target.cpu().clone().detach().squeeze(0)
    output_image = transforms.ToPILImage()(output_image)
    output_image.save(output_path)

# Example usage
if __name__ == "__main__":
    neural_style_transfer('product.jpg', 'clothing.jpg', 'result_nst.jpg')