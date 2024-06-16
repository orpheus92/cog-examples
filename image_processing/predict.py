from typing import Any
import cv2
import json
import numpy as np
from ultralytics import YOLO
from cog import BasePredictor, Input, Path
from PIL import Image
from openai import OpenAI
import base64
import pandas as pd
from transformers import pipeline
import requests
import math
import torch

# from tensorflow.keras.applications.resnet50 import (
#     ResNet50,
#     decode_predictions,
#     preprocess_input,
# )
# from tensorflow.keras.preprocessing import image as keras_image

YOLOModel = "best.pt"
category_file = "categories.json"
confidence_cut = 0.4
#API_KEY_GPT = ''
MODEL = "gpt-4o"
DB1 = 'Ikea_with_shape.json'
DB2 = 'crate_barrel_with_shape.csv'
map_dict = {'major':'name','minor':['color','texture','shape']}
FOV = 72
TRAN_MATRIX = np.array([
[0.99414575, 0.047715146, 0.09693929, 0.0],
[-0.03906529,0.995243,-0.08924736,0.0],
[-0.10073663,0.08493792, 0.9912873,0.0],
[-0.6388,0.09999962,0.37271586,0.9999999]
])

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.img2obj = {}
        self.classes = {}
        self.cat_mapping = {}

        # Load Model for Object Detection
        self.model = YOLO(YOLOModel)
        
        # Load super category
        self.supercat = self.load_superCat(category_file)

        # Load GPT_MODEL
        self.API_KEY_GPT = API_KEY_GPT
        self.text_model = MODEL
        self.client = OpenAI(api_key=self.API_KEY_GPT)

        # Load Furniture DB 
        # To do: Fix this when db is more generalizable
        self.load_furnitureDB(DB1, DB2)

    def image_resize(self, image, width=640):

        # Open the original image
        original_image = Image.open(image)
        aspect_ratio = original_image.width / original_image.height

        # Resize the image while preserving aspect ratio
        # To do: Resizing for image with height>width
        new_width = width
        new_height = int(width / aspect_ratio)

        # Resize the image
        resized_image = original_image.resize((new_width, new_height))
        return resized_image

    def object_detection(self, image):
        preds = self.model.predict(image)
        preds[0].save(filename='detect_out.png')  # save to disk for now
        return preds
    
    def load_superCat(self, file_path):
        with open(file_path, 'r') as file:
            supercat = json.load(file)
        
        return supercat
        
    def process_detection(self, detect_result, conf_cut=0, intersection_area=0.5):

        self.img2obj['obj'] = []
        self.img2obj['conf'] = []
        self.img2obj['path'] = []
        self.img2obj['range'] = []

        classes = detect_result.names.copy()
        for key, value in classes.items():
            classes[key] = ' OR '.join(value.split('/'))

        # Hard coded due to the category names in YOLO
        classes[6] = 'Corner Table OR Side Table'
        classes[3] = 'Bookcase OR Jewelry Armoire'
        classes[26] = 'Three-seat Sofa OR Multi-seat Sofa'

        self.classes = classes

        # Mapping from cat to super_cat
        cat_mapping = {}
        for cat in self.supercat:
            cat_mapping[classes[cat['id']]] = ' OR '.join(cat['super-category'].split('/'))
        cat_mapping['Pendant Lamp'] = 'Lamp'
        cat_mapping['Ceiling Lamp'] = 'Lamp'

        self.cat_mapping = cat_mapping

        num_obj = len(detect_result.boxes.cls[detect_result.boxes.conf>conf_cut])
        # print("Object number above confidence: ", num_obj)

        coords = detect_result.boxes.data[detect_result.boxes.conf>conf_cut, :4]
        # print(coords.shape)
    
        overlaps = []
        for i in range(coords.size(0)):
            for j in range(i + 1, coords.size(0)):
                area_i = self._area(*coords[i])
                area_j = self._area(*coords[j])
                inter_area = self._intersection_area(coords[i], coords[j])
                if inter_area / area_i > intersection_area or inter_area / area_j > intersection_area:
                    overlaps.append((i, j))
        # print('overlap:', overlaps)

        removeidx = []
        for _idx in overlaps:
            removeidx.append(_idx[1])

        self.img2obj['rm_idx'] = removeidx
        # print('index to remove:', removeidx)

        for idx2 in range(num_obj):#len(detect_result.boxes.cls)):
            if idx2 not in removeidx:
                obj = classes[int(detect_result.boxes.cls[idx2])]
                self.img2obj['obj'].append(obj)
                conf = detect_result.boxes.conf[idx2]
                self.img2obj['conf'].append(conf)
                x0, y0, x1, y1 = [int(ts) for ts in detect_result.boxes.data[idx2][:4]]
                self.img2obj['range'].append([x0, y0, x1, y1])
                selected_region = detect_result.orig_img[y0:y1, x0:x1]
                
                # This might be changed if data cannot be saved 
                sel_path = '{}_{}.png'.format(idx2, obj)
                
                self.img2obj['path'].append(sel_path)
                cv2.imwrite(sel_path, selected_region)

        return 

    def desc_gen(self):

        # for image_path in img2obj:
        self.img2obj['desc'] = []
        for idxxx, obj_path in enumerate(self.img2obj['path']):
            objlist = ",".join(format(self.img2obj['obj'][idxxx]).split(' OR '))
            objlist = "[{}]".format(objlist)
            #print('Detected furnitures: ', objlist)
            if self.API_KEY_GPT:

                response_text = self._generate_desc(obj_path, objlist)
                # print(objlist, response_text)
                if response_text is not None:
                    self.img2obj['desc'].append(response_text)
                else:
                    print("No Response from GPT.")
            else:
                print("GPT API key not found.")

        return 

    def load_furnitureDB(self, db1, db2):
        file_path = db1
        # Open and read the JSON file
        with open(file_path, 'r') as file:
            furnitureDB = json.load(file)

        count = 0
        filtered_list = []
        for d in furnitureDB['data']:
            curdict = {}
            for kkk in ["name", "id", "thumbnailUrl"]:
                curdict[kkk] = d[kkk].lower()
            if "gpt_desc" in d:
                curdesc = d["gpt_desc"]['furniture']
                if type(curdesc)==dict:
                    count+=1
                    for kkk in [ "type", "size", "shape", "texture", "color"]:
                        if kkk in curdesc:
                            curdict[kkk] = str(curdesc[kkk]).lower()
                    filtered_list.append(curdict)

        # Create a DataFrame from the filtered data
        dfDB = pd.DataFrame(filtered_list)

        db2_ = pd.read_csv(db2)

        newDB = pd.DataFrame(columns = ['name', 'id', 'thumbnailUrl', 'size', 'shape', 'texture', 'color', 'type'])

        newDB[['name', 'id', 'thumbnailUrl', 'size', 'shape', 'texture', 'color',
            'type']] = db2_[db2_.contains_furniture][['name','id', 'thumbnail_url','size', 'shape', 'texture', 'color','furniture_name']]

        for k in newDB:
            if k!='thumbnailUrl':
                newDB[k] =newDB[k].str.lower()

        self.furnitureDB = pd.concat([dfDB,newDB]).reset_index(drop=True)

        return 

    def find_matched_furnitures(self, map_dict):
        self.img2obj['foundmatch_id'] = []
        self.img2obj['foundmatch_url'] = []
        for curdata in self.img2obj['desc']:
            #out_list.append(self.furnitureDB[])
            res = self._find_matching(curdata, map_dict)
            self.img2obj['foundmatch_id'].append(self.furnitureDB[res]['id'])
            self.img2obj['foundmatch_url'].append(self.furnitureDB[res]['thumbnailUrl'])
        return #out_list

    def depth_estimation(self, image):

        pipe = pipeline(task="depth-estimation", model="Intel/dpt-large")
        #image = Image.open(resized_files[0])
        result = pipe(image)
        return result


    def depth_to_3d(self, depth_image):#, intrinsic_params, camera_pose):

        fx, fy, cx, cy = self._get_intrinsic(depth_image)

        translation, rotation, scale_, perspective_ = self._get_extrinsic()

        depth_image = self._invert_depth(depth_image)

        # Create grid of pixel coordinates
        u, v = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))

        # Compute 3D coordinates in camera coordinates
        Z = depth_image
        X_cam = (u - cx) * Z / fx
        Y_cam = (v - cy) * Z / fy

        self.coordinates_3d = self._transform_coordinates(np.dstack((X_cam, Y_cam, Z)), rotation, translation)

        return 

    def comp_coords(self):
        self.img2obj['fur_coords_3d'] = []#obj_coord
        self.img2obj['fur_size_3d'] = []#f_size
        #obj_name = []
        for idxx, d in enumerate(self.img2obj['range']):#detection_results[0].boxes.data[detection_results[0].boxes.conf>.36, :]:
            x0, y0, x1, y1 = d
            coordinates = self.coordinates_3d[y0:y1, x0:x1,:]
            x_mean = (np.max(coordinates[:,:,0])+np.min(coordinates[:,:,0]))/2
            y_mean = (np.max(coordinates[:,:,1])+np.min(coordinates[:,:,1]))/2
            z_mean = (np.max(coordinates[:,:,2])+np.min(coordinates[:,:,2]))/2
            width = np.max(coordinates[:,:,0])-np.min(coordinates[:,:,0])
            height = np.max(coordinates[:,:,1])-np.min(coordinates[:,:,1])
            length = np.max(coordinates[:,:,2])-np.min(coordinates[:,:,2])
            self.img2obj['fur_coords_3d'].append([x_mean, y_mean, z_mean])
            self.img2obj['fur_size_3d'].append([width, height, length])
            #obj_name.append(self.img2obj['range'][idxx])
        
        # obj_coord = np.array(obj_coord)
        # f_size = []
        # for objs in obj_size:
        #     f_size.append(math.sqrt(math.pow(objs[0], 2) + math.pow(objs[2], 2)))
        
        #self.obj_name = obj_name
        #self.img2obj['fur_coords_3d'] = obj_coord
        #self.img2obj['fur_size_3d'] = f_size
        return 

    def prepare_output(self):#, match):
        curout = {}
        for j_, obj in enumerate(self.img2obj['obj']):
            curout[j_] = {}
            curout[j_]['obj'] = obj
            curout[j_]['id_found'] = self.img2obj['foundmatch_id'][j_].values.tolist()
            curout[j_]['3d_coords'] = self.img2obj['fur_coords_3d'][j_]
        return curout

    def _get_intrinsic(self, depth_image):

        sensor_width_pixels = depth_image['depth'].size[0]#depth_image.shape[1] #640#, 4000  # Width of the image sensor in pixels
        sensor_height_pixels = depth_image['depth'].size[1]#depth_image.shape[0]#358# 3000  # Width of the image sensor in pixels

        fov_degrees = FOV  # Field of view in degrees

        # Convert FOV from degrees to radians
        fov_radians = math.radians(fov_degrees)

        focal_length_width_pixels = self._focal_length_pixels(sensor_width_pixels, fov_radians)
        focal_length_height_pixels = self._focal_length_pixels(sensor_height_pixels, fov_radians)

        #fx, fy = focal_length_width_pixels, focal_length_height_pixels
        cx, cy = depth_image['depth'].size[0]/2, depth_image['depth'].size[1]/2

        return (focal_length_width_pixels, focal_length_height_pixels, cx, cy)

    def _get_extrinsic(self):
        matrix = TRAN_MATRIX
        """Decompose a 4x4 perspective transformation matrix into its components."""
        # Ensure the matrix is a 4x4 matrix
        assert matrix.shape == (4, 4), "The input matrix must be 4x4."
        
        # Normalize the matrix to ensure the bottom-right element is 1
        matrix /= matrix[3, 3]
        
        # Extract the translation
        translation = matrix[:3, 3]
        
        # Extract the perspective projection (3rd row, 4th column)
        perspective = matrix[3, :3]
        
        # Remove the perspective from the matrix to get the affine part
        affine_matrix = matrix.copy()
        affine_matrix[3, :] = [0, 0, 0, 1]
        
        # Extract the scale factors
        scale_x = np.linalg.norm(affine_matrix[:3, 0])
        scale_y = np.linalg.norm(affine_matrix[:3, 1])
        scale_z = np.linalg.norm(affine_matrix[:3, 2])
        scale = np.array([scale_x, scale_y, scale_z])
        
        # Remove the scale from the matrix to get the rotation matrix
        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[:, 0] = affine_matrix[:3, 0] / scale_x
        rotation_matrix[:, 1] = affine_matrix[:3, 1] / scale_y
        rotation_matrix[:, 2] = affine_matrix[:3, 2] / scale_z

        return translation, rotation_matrix, scale, perspective

    def _invert_depth(self, depth_image):

        predicted_depth = depth_image['predicted_depth']
        # Interpolate the predicted depth values to the original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=depth_image['depth'].size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()

        formatted = (output * 255 / np.max(output)).astype("uint8")# normalized_depth#((np.max(output)-output) * 255 / np.max(output)).astype('uint8')
        formatted = np.max(formatted)-formatted
        return formatted

    def _transform_coordinates(self, coordinates_3d, rotation_matrix, translation_vector):
        # Reshape the coordinates to (3, -1) for matrix multiplication
        h, w, _ = coordinates_3d.shape
        coordinates_3d_flat = coordinates_3d.reshape(-1, 3).T  # Shape: (3, height * width)
        
        # Apply the rotation and translation
        transformed_coordinates_flat = rotation_matrix @ coordinates_3d_flat + translation_vector[:, np.newaxis]
        
        # Reshape back to (height, width, 3)
        transformed_coordinates_3d = transformed_coordinates_flat.T.reshape(h, w, 3)
        
        return transformed_coordinates_3d

    def _focal_length_pixels(self, sensor_dimension_pixels, fov):
        focal_length_pixels = sensor_dimension_pixels / (2 * math.tan(fov / 2))
        return focal_length_pixels

    def _find_matching(self, curdata, map_dict, fur_name='furniture_name'):

        db = self.furnitureDB
        majorname = map_dict['major']
        # Use cat or super cat
        # print(curdata)
        cur_cat = curdata[fur_name].lower()
        # if can't find using current cat
        if not db[majorname].str.contains(cur_cat).any():
            # Use super cat
            print('Try to find super cat for: ', curdata[fur_name])
            for k in self.cat_mapping:
                if curdata[fur_name] in k:
                    supercats = self.cat_mapping[k].split(' OR ')#[0]
                    for cat in supercats:
                        curfind = (db[majorname].str.contains(cat.lower())).any()#(~(db[majorname].str.contains(cat.lower()))).all()
                        if curfind:
                            cur_cat = cat.lower()
                            print('found supercat: ', cur_cat)
                            break
        if not db[majorname].str.contains(cur_cat).any():# (~(db[majorname].str.contains(cur_cat))).all():
            print('No Matching Found for Name: ', cur_cat)
            # Add return or False Statement
            return db[majorname].str.contains(cur_cat)
        prev_found = db[majorname].str.contains(cur_cat)
        for feature in map_dict['minor']:
            detect_v = curdata[feature]
            cur_found = db[feature].str.contains(detect_v)
            if (~(prev_found&cur_found)).all(): # No matching Found
                return prev_found
            else:
                prev_found = (prev_found&cur_found)
                print('Used feature: ', feature)
        return prev_found

    def _area(self, x0, y0, x1, y1):
        return (x1 - x0) * (y1 - y0)

    # Function to calculate the intersection area of two rectangles
    def _intersection_area(self, rect1, rect2):
        x0 = max(rect1[0], rect2[0])
        y0 = max(rect1[1], rect2[1])
        x1 = min(rect1[2], rect2[2])
        y1 = min(rect1[3], rect2[3])

        if x0 < x1 and y0 < y1:
            return (x1 - x0) * (y1 - y0)
        else:
            return 0


    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _generate_desc(self, image_path, furniture_list):

        base64_image = self._encode_image(image_path)

        TextInput = 'Could you find any of the furnitures listed in the \
        Furniture List {} in the picture? \
        Ensure the furniture name exactly matches the one in the list.  \
        Only find one furniture with its name, size, shape, texture, color. \
        For the size, estimate the width, length, and height.\
        Respond in JSON Format that only contains furniture_name, size, shape, texture, color. \
        If you could not find the furniture from the list,\
        provide a furniture name for the object in the picture and respond \
        in a similar way with an additional key new = 1 in the JSON .'.format(furniture_list)

        response = self.client.chat.completions.create(
            model=self.text_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that responds \
                in JSON format. Help me describe the furniture in the picture!"},
                {"role": "user", "content": [
                    {"type": "text", "text": "{}".format(TextInput)},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            ],
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content[8:-4])

    # Define the arguments and types the model takes as input
    def predict(self, image: Path = Input(description="Image to detect")) -> Any:
        """Run a single prediction on the model"""

        # Resize image
        resized_img = self.image_resize(image)

        # Object Detection
        preds = self.object_detection(resized_img)

        # Furniture Selection
        self.process_detection(preds[0], confidence_cut)

        # Description Generation
        self.desc_gen()

        #print(self.furnitureDB.columns, self.furnitureDB.shape)

        #print(self.classes, self.supercat)

        # print(self.img2obj)
        self.find_matched_furnitures(map_dict)
        #print(matched)

        # depth_image
        depth_img = self.depth_estimation(Image.fromarray(preds[0].orig_img))
        depth_img["depth"].save('depth_img.png')

        self.depth_to_3d(depth_img)

        self.comp_coords()
        #print(self.img2obj)
        outdict = self.prepare_output()#matched)

        #classes = json.dumps(preds[0].names)
        #data = preds[0].boxes.data.numpy()
        out = json.dumps(outdict)



        # Return the top 3 predictions
        # print(classes, data)
        return out# classes, data # decode_predictions(preds, top=3)[0]
