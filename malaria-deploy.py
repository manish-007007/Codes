import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision import transforms , models
from PIL import Image
import matplotlib.pyplot as plt


def main():
    colorr=""" <style>
     body{
     background-color:rgb(209, 228, 37);
     }
   
    </style> """
    st.markdown(colorr,unsafe_allow_html=True)

    ## Transformation
    trans = transforms.Compose([
        transforms.Resize(224) , transforms.CenterCrop(224)  , transforms.RandomHorizontalFlip(p= 0.4) ,
        transforms.RandomVerticalFlip(p=0.5)  ,transforms.ToTensor() ])

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if  train_on_gpu:
        print('CUDA is  available.  Training on GPU ...')
        device = "cuda"
    else:
        print('CUDA is not available!  Training on CPU ...')
        device = "cpu"

    ## Use transfer learning (model = vgg16)
    model = models.resnet152(pretrained = True)
    model.to(device)
    ## freeze params
    for param in model.fc.parameters():
        param.required_grad = False
    ## change the output layer
    num_ftrs = model.fc.in_features
    out = 2     #( pred class)
    model.fc = nn.Linear(num_ftrs, out)



    model = model.to(device)
    pickle = r"C:\Users\MANISH SHARMA\myjupcodes\malaria_resnet152.pt"
    model.load_state_dict(torch.load(pickle))
    model.cuda()


    ## test with your own image
    ## test with your own image

    model.eval()
    type_of_files=['png','jpg','jpeg']
    #st.set_option('deprecation.showfileUploaderEncoding', False)
    images=st.file_uploader("Upload",type=type_of_files)
    img=Image.open(images)
    st.image(img)
    #img_name = r"C:\Users\MANISH SHARMA\Desktop\deeplearning-pytorch\malaria\not-infect1.jpg" # change this to the name of your image file.
    def predict_image(image_path, model):
        image = Image.open(image_path)
        image_tensor = trans(image)
        image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(device)
        print(image_tensor.shape)
        
        output = model(image_tensor)
        index = output.argmax().item()
        if index == 0:
            st.write( ''' Non-Parasitic ''')
        elif index == 1:
            st.write(''' Parasitic ''')
        
    predict_image(images,model)

if __name__ == "__main__":
    main()