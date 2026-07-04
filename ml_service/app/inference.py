import torch
import numpy as np
import cv2
import albumentations as A
import segmentation_models_pytorch as smp
from PIL import Image
import torchvision.models as models
import torch.nn as nn

class TalcPredictor:
    def __init__(self, weights_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=2
        )
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def predict_panorama(self, image_np, patch_size=512, stride=512):
        h_orig, w_orig, c = image_np.shape
        
        pad_y = max(0, patch_size - h_orig)
        pad_x = max(0, patch_size - w_orig)
        if pad_y > 0 or pad_x > 0:
            image_np = cv2.copyMakeBorder(image_np, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)
            
        h, w, c = image_np.shape
        
        full_mask = np.zeros((h, w), dtype=np.uint8)
        
        y_coords = list(range(0, h - patch_size + 1, stride))
        if y_coords[-1] + patch_size < h:
            y_coords.append(h - patch_size)
            
        x_coords = list(range(0, w - patch_size + 1, stride))
        if x_coords[-1] + patch_size < w:
            x_coords.append(w - patch_size)

        for y in y_coords:
            for x in x_coords:
                patch = image_np[y:y+patch_size, x:x+patch_size]
                
                augmented = self.transform(image=patch)
                patch_tensor = torch.from_numpy(augmented['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
                
                device_type = "cuda" if "cuda" in str(self.device) else "cpu"
                with torch.inference_mode():
                    with torch.amp.autocast(device_type):
                        preds = self.model(patch_tensor)
                    preds_class = preds.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                
                full_mask[y:y+patch_size, x:x+patch_size] = np.maximum(
                    full_mask[y:y+patch_size, x:x+patch_size], 
                    preds_class
                )

        cropped_mask = full_mask[:h_orig, :w_orig]
        talc_pixels = np.sum(cropped_mask == 1)
        total_pixels = h_orig * w_orig
        talc_percentage = (talc_pixels / total_pixels) * 100
        
        orig_img = image_np[:h_orig, :w_orig].copy()
        overlay = orig_img.copy()
        overlay[cropped_mask == 1] = [0, 120, 255]
        overlay_img = cv2.addWeighted(overlay, 0.4, orig_img, 0.6, 0)
        
        return {
            "talc_percentage": round(talc_percentage, 2),
            "overlay": Image.fromarray(overlay_img),
            "mask": Image.fromarray((cropped_mask * 255).astype(np.uint8))
        }


class ClassPredictor():
    def __init__(self, weights_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['ordinary', 'talcose', 'refractory']
        # Рядовая - 0, тальковая - 1, труднообогатимая (упорная) - 2

        self.model = models.efficientnet_b3(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 3)

        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = A.Compose([
            A.Resize(300, 300),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def predict(self, image_np, patch_size=512, stride=512):
        "Нарезаем панораму, классифицируем патчи. Возвращаем средний класс"

        h_orig, w_orig, c = image_np.shape

        pad_y = max(0, patch_size - h_orig)
        pad_x = max(0, patch_size - w_orig)

        if pad_y > 0 or pad_x > 0:
            image_np = cv2.copyMakeBorder(image_np, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)

        h, w, c = image_np.shape
        
        predictions = []
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image_np[y:y+patch_size, x:x+patch_size]
                augmented = self.transform(image=patch)
                patch_tensor = torch.from_numpy(augmented['image'].transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    with torch.amp.autocast(device_type):
                        preds = self.model(patch_tensor)
                # Получаем вероятности классов через Softmax и переводим в Float32 перед numpy
                probs = torch.softmax(preds, dim=1).squeeze(0).float().cpu().numpy()
                predictions.append(probs)

        mean_probs = np.mean(predictions, axis=0)
        best_class_idx = np.argmax(mean_probs)
        predicted_class = self.classes[best_class_idx]
        
        return {
            #рядовая- 0, тальковая - 1, трудообогатимая - 2
            "class": predicted_class,
            "probabilities": {self.classes[i]: float(mean_probs[i]) for i in range(3)}
        }


#ИСПОЛЬЗОВАНИЕ 2Х КЛАССОВ

# if __name__ == '__main__':
#     class_predictor = ClassPredictor("checkpoints_cls/best_classifier_weights.pt")
#     talc_predictor = TalcPredictor("best_unet_talc_f1_0_898.pth")
#
#     img_path = "dataset/ores_by_grade/talcose/DSCN5186.JPG"
#     img = np.array(Image.open(img_path).convert("RGB"))
#
#     class_result = class_predictor.predict(img)
#     print(f"Класс руды: {class_result['class']}")
#
#     seg_result = talc_predictor.predict_panorama(img)
#     print(f"Процент талька: {seg_result['talc_percentage']}%")
#         
#     seg_result['overlay'].save("test_pipeline_overlay.png")
#     seg_result['mask'].save("test_pipeline_mask.png")
#     print("Маски успешно сохранены!")
