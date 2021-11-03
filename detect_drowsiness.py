"""
This files applies the trained drowsy detector to videos and images.
"""

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
import cv2
import time
from argparse import ArgumentParser



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', required=True, help='Filepath to the pickled model state')
    parser.add_argument('--on-gpu', action='store_true', default=False, help='Set True for running inference on gpu')
    args = parser.parse_args()

    # labels
    classes = ['awake', 'background', 'drowsy']

    # load model skeleton
    model = models.mobilenet.mobilenet_v2()
    
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(classes))

    # load model based on device type
    if args.on_gpu:
        print('using gpu')
        device = torch.device("cuda")
        model.load_state_dict(torch.load(args.model))
        model = model.to(device)
    else:
        print('using cpu')
        device = torch.device("cpu")
        model.load_state_dict(torch.load(args.model, map_location=device))

    model.eval()

    # preprocessing image
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.Grayscale(3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Start inference
    cap = cv2.VideoCapture(0)
    with torch.no_grad():    
        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret:
                # preprocessing frame
                img = torch.from_numpy(frame)
                img = img.permute(2, 0, 1) # (H, W, C) -> (C, H, W), channel last -> channel first
                img = preprocess(img).to(device)
                img.unsqueeze_(0)
                
                # make detection
                output = model(img)
                _, preds = torch.max(output, 1)

                print(classes[preds]) 
                is_drowsy = 'Drowsiness detected!' if classes[preds] == 'drowsy' else 'No drowsiness detected'

                cv2.putText(frame, is_drowsy, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.imshow('Detect Drowsiness', frame)

                time.sleep(0.1)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
                
    cap.release()
    cv2.destroyAllWindows()
