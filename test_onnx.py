import onnx
import onnxruntime
from torchvision import transforms
from PIL import Image
import numpy as np


# def preprocess_numpy(img):
#     if img.shape[2] == 3 and img.shape[2] != 1:
#         img = cv2.cvtColor

def preprocess_numpy(img):
    resize = transforms.Resize((224, 224))   #must same as here
    crop = transforms.CenterCrop((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = resize(img)
    img = crop(img)
    img = to_tensor(img)
    img = normalize(img)
    return img



# mtailr = onnx.load("Converted_pytorch_model_weights.onnx")
ort_session = onnxruntime.InferenceSession("Converted_pytorch_model_weights.onnx")
ort_session.get_modelmeta()
first_input_name = ort_session.get_inputs()[0].name
first_output_name = ort_session.get_outputs()[0].name

img1 = Image.open("n01667114_mud_turtle.JPEG")
img2 = Image.open("n01440764_tench.jpeg")

image1 = preprocess_numpy(img1)
image2 = preprocess_numpy(img2)
outputs = ort_session.run([first_output_name],{first_input_name:[np.array(image2)]})
print(np.argmax(outputs))
outputs = ort_session.run([first_output_name],{first_input_name:[np.array(image1)]})
print(np.argmax(outputs))

