from transformers import pipeline
# import onnxruntime
import torch
import model 
from PIL import Image
# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global ort_session
    ort_session = model.load_model("Converted_pytorch_model_weights.onnx")
    # ort_session = onnxruntime.InferenceSession("Converted_pytorch_model_weights.onnx")

    
    device = 0 if torch.cuda.is_available() else -1

    # model = pipeline('fill-mask', model='bert-base-uncased', device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(pathofimg):
    global ort_session
    img = Image.open(pathofimg)
    # img = model.preprocess_numpy(img)

    result = model.inference(img,ort_session)
    # Parse out your arguments
    # prompt = model_inputs.get('prompt', None)
    # if prompt == None:
    #     return {'message': "No prompt provided"}
    
    # Run the model
    # result = model(prompt)

    # Return the results as a dictionary
    return result
# init()
# retee = inference("./n01667114_mud_turtle.JPEG")
# print(retee)