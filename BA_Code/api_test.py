import numpy as np
import json
import pandas as pd
import requests
import requests
import base64
import io
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

def csvtojson():

    with open("BA_Code\API_Images\image_counter.json", "r") as json_file:
        my_dict = json.load(json_file)  
                    
    path = 'BA_Code\API_Annotations\dataframe_anno.csv' # the path to the CSV file
    save_json_path = 'BA_Code\API_Annotations\coco_annotations.coco{}.json'.format(my_dict[0]["img_counter"])
    #save_json_path = 'BA_Code\API_Annotations\coco_annotations.coco.json'


    data = pd.read_csv(path)

    images = []
    categories = []
    annotations = []

    category = {}
    category["supercategory"] = 'none'
    category["id"] = 0
    category["name"] = 'None'
    categories.append(category)

    data['fileid'] = data['filename'].astype('category').cat.codes
    data['categoryid']= pd.Categorical(data['class'],ordered= True).codes
    data['categoryid'] = data['categoryid']+1
    data['annid'] = data.index

    def image(row):
        image = {}
        image["height"] = row.height
        image["width"] = row.width
        image["id"] = row.fileid
        image["file_name"] = row.filename
        return image

    def category(row):
        category = {}
        category["supercategory"] = 'None'
        category["id"] = row.categoryid
        category["name"] = row[2]
        return category

    def annotation(row):
        annotation = {}
        area = (row.xmax -row.xmin)*(row.ymax - row.ymin)
        annotation["segmentation"] = []
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = row.fileid

        annotation["bbox"] = [row.xmin, row.ymin, row.xmax -row.xmin,row.ymax-row.ymin ]

        annotation["category_id"] = row.categoryid
        annotation["id"] = row.annid
        return annotation

    for row in data.itertuples():
        annotations.append(annotation(row))

    imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
    for row in imagedf.itertuples():
        images.append(image(row))

    catdf = data.drop_duplicates(subset=['categoryid']).sort_values(by='categoryid')
    for row in catdf.itertuples():
        categories.append(category(row))

    data_coco = {}
    data_coco["images"] = images
    data_coco["categories"] = categories
    data_coco["annotations"] = annotations
    json.dump(data_coco, open(save_json_path, "w"), indent=4)

def readanno():
  
    annotation_filename = "BA_Code\API_Annotations\dataframe_anno.csv"

    # Read Annotation as String
    annotation_str = open(annotation_filename, "r").read()

    # Construct the URL
    upload_url = "".join([
        "https://api.roboflow.com/dataset/rico-dataset/annotate/abc123",
        "?api_key=bJkakjWyse01WesCPFUH",
        "&name=", annotation_filename
    ])

    # POST to the API
    r = requests.post(upload_url, data=annotation_str, headers={
        "Content-Type": "text/plain"
    })

    # Output result
    print(r.json())

def upload_image(img_path):
    # Load Image with PIL
    image = Image.open(img_path).convert("RGB")

    # Convert to JPEG Buffer
    buffered = io.BytesIO()
    image.save(buffered, quality=90, format="JPEG")

    # Construct the URL
    upload_url = "".join([
        "https://api.roboflow.com/dataset/android-ui-objects/upload",
        "?api_key=bJkakjWyse01WesCPFUH"
    ])

    m = MultipartEncoder(fields={'file': (img_path, buffered.getvalue(), "image/jpeg")})
    r = requests.post(upload_url, data=m, headers={'Content-Type': m.content_type})

    # Output result
    #print(r.json()["id"])
    return r.json()["id"]

def upload_annotation(img_id):

    annotation_filename = "BA_Code\API_Annotations\coco_annotations.coco.json"

    
    # Read Annotation as String
    annotation_str = open(annotation_filename, "r").read()

    # Construct the URL
    upload_url = "".join([
        "https://api.roboflow.com/dataset/anki-vector-customobjectcodes/annotate/" + img_id,
        "?api_key=bJkakjWyse01WesCPFUH",
        "&name=", annotation_filename
    ])

    # POST to the API
    r = requests.post(upload_url, data=annotation_str, headers={
        "Content-Type": "text/plain"
    })

    # Output result
    print(r.json())

