# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
input_image_path = "n01667114_mud_turtle.JPEG"
model_inputs = {'input_image': "n01667114_mud_turtle.JPEG"}

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.text)