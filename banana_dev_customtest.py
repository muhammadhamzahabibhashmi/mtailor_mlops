import banana_dev as banana

api_key = "492659ed-745b-40d5-94cd-92b48f7a1ffb"
model_key = "f4c07e49-71cb-46a5-97a5-4bed79ffbeb0"

input_image_path = "n01667114_mud_turtle.JPEG"
model_inputs = {'input_image': "n01667114_mud_turtle.JPEG"}


# model_inputs = {YOUR_MODEL_INPUT_JSON} # anything you want to send to your model

out = banana.run(api_key, model_key, model_inputs)
print(out)