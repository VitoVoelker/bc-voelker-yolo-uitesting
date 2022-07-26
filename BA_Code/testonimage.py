from PIL import Image
import torch

model = torch.hub.load('yolov5', 'custom', path='Training\Run9_500_UIObj\\best (7).pt', source='local', force_reload=True)  # local repo

#open image 
imgs = Image.open('Testimages\978064ec-623d-4091-b4be-1e69a92e22bb.jpg') 

results = model(imgs, size=640) 


results.save('Testimages\\results') 