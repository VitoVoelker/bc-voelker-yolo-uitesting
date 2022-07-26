import imgaug.augmenters as iaa
import cv2
import glob

# 1. Load Dataset
images = []
images_path = glob.glob("BA_Code\ImageAugmentation\Origininals\\train\*.jpg")
for img_path in images_path:
    img = cv2.imread(img_path)
    images.append(img)

# 2. Image Augmentation
augmentation = iaa.Sequential([
     
    iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),
        
])    

# 3. Show Images
while True:
    augmented_images = augmentation(images=images)
    for img in augmented_images:
        cv2.imshow("Image", img)
        cv2.waitKey(0)