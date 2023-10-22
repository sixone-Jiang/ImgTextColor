# use flask create a server 
# input is an image, output is the result of ocr
from fastapi import FastAPI, File, UploadFile
from starlette.responses import FileResponse

import cv2
from rec_image_color import rec_an_image_color, rec_an_image_color_cv2, rec_an_image_color_cv2_v2
import base64
import numpy as np

app = FastAPI()

def base64_to_image(base64_str):
    imgString = base64.b64decode(base64_str)
    nparr = np.frombuffer(imgString, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.post("/rec_image_color_upload_show_demo")
async def rec_image_color_upload_show_demo(file: UploadFile = File(...)):
    try:
        file_path = './upload_image/' + file.filename
            
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        #rec_an_image_color(file_path, show_demo=True)
        cv2_image = cv2.imread(file_path)
        rec_an_image_color_cv2_v2(cv2_image, show_demo=True)

        
        return FileResponse(
            './upload_image/result.jpg',
            filename='demo_show.jpg',
        )
    except Exception as e:
        return {"message": str(e)}

@app.post("/rec_image_color_base64_image_list")
def rec_image_color_base64_image_list(base64_images: list[str]):
    result_color_list = []
    for base64_image in base64_images:
        cv2_image = base64_to_image(base64_image)
        result_color = rec_an_image_color_cv2_v2(cv2_image, show_demo=False)
        print(result_color)
        result_color_list.append(result_color)
    
    return result_color_list