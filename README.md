**Run Deliverables**
    To setup environment RUN "pip3 install -r requirements.txt"  <br />

    convert_to_onnx.py | 
        RUN: "python convert_to_onnx.py" you can change the pretrained model e.g: pytorch_model_weights.pth to your own model and file with onnx  extension will be saved   
    test_onnx.py |
        RUN: "python test_onnx.py" have to read two images in the code and it will classify them into the classes its trained on
    model.py |
        RUN: "python model.py" It has 3 functions we have to call load_model and inference and it will classify them as it is done in test_onnx.py file a bit generic
    test_server.py | 
        RUN: "python test_server.py" It is the API server code which reads image path and calls the above functions present in model.py 
    banana_dev_customtest.py 
        RUN "python banana_dev_customtest.py" It is sdk created for my deployed model