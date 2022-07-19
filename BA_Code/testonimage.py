from PIL import Image
import torch

model = torch.hub.load('yolov5', 'custom', path='Training\Run6_500\\best.pt', source='local', force_reload=True)  # local repo

#open image 
imgs = Image.open('Testimages\\0f9724d5-ea77-4a47-92b5-744965915416.jpg') 

results = model(imgs, size=640) 


results.save('Testimages\\results') 