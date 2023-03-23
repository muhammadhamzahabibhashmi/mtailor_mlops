import onnxruntime
from torchvision import transforms
# from PIL import Image
import numpy as np

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

def load_model(onnxpath):
    ort_session = onnxruntime.InferenceSession(onnxpath)
    return ort_session

def inference(img,model):
    model.get_modelmeta()
    first_input_name = model.get_inputs()[0].name
    first_output_name = model.get_outputs()[0].name
    image = preprocess_numpy(img)
    outputs = model.run([first_output_name],{first_input_name:[np.array(image)]})
    return np.argmax(outputs)

