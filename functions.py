import os


# for converting the frames into bytes
import cv2 

# and for processing arrays   
import numpy as np

# for encoding and decoding Custom Vision predictions 
import json

# for converting the Custom Vision predictions to dataframe   
import pandas as pd

# import async packages
import asyncio
import aiohttp

# for file name pattern matching   
import fnmatch  

# for displaying images from the processes output video   
import matplotlib.pyplot as plt

# importing other required libraries
import random
import textwrap
import datetime 
from PIL import Image
import time 
import GlobalVariables


# web service end-point for the Custom Vision model    
# we will process video frames (which are images)   
POST_URL = "Your custom vision model endpoint"

# providing prediction key
HEADERS = {'Prediction-Key': "Your custom vision model prediction key","Content-Type":"application/json"}

# number of API calls per pool of request   
MAX_CONNECTIONS = 100 
# initializing the height and width for frames in the video 
WIDTH = 0
HEIGHT = 0

# returns frame array of video
def getFrames(path):
    images = []
    byteImages = []
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            images.append(image)
            count += 1
    global HEIGHT, WIDTH
    HEIGHT, WIDTH, lay = images[0].shape
    return images




# Returns ByteImages for POST request params
def convertCVImagesToByteImages(frames):

    toReturnByteImages = []
    for frame in frames:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        if is_success:
            byte_im = im_buf_arr.tobytes()
            toReturnByteImages.append(byte_im)
    return toReturnByteImages




# making async requests call
async def fetch(session, url, data,i):
    print("Request #" + str(i) + " Sent")
    async with session.post(url,headers=HEADERS, data=data) as response:
        resp = await response.text()
        print("Request #"+ str(i) + " Complete")
        return resp

# Setting up for batch requests
async def fetch_all(byteImages):
    conn = aiohttp.TCPConnector(limit=len(byteImages)+1)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        i = 0
        for image in byteImages:
            i = i+1
            # if i%20 == 0:
                # time.sleep(3)
            tasks.append( fetch(session, POST_URL, image, i) )

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        return responses


# defining a function to convert raw JSON to a dataframe
def getDFfromResponse(response,threshold):
    resp = json.loads(response)
    pred= resp['predictions']
    pred_df=pd.DataFrame(pred)
    resp_df = pred_df[pred_df['probability'] > float(threshold)]
    return resp_df

# defining a function to converting the bounding box dataframe to a list 
def boundingBoxListFromDF(df):
    cood_list = df['boundingBox'].tolist()
    return cood_list

# defining a function for adjusting the size of bounding box 
def fitAccordingToSize(coods):
    print("Multiplying by " + str(WIDTH) + " , " + str(HEIGHT))
    for i in range(len(coods)):
        coods[i]['left']=coods[i]['left']*WIDTH
        coods[i]['top']=coods[i]['top']*HEIGHT
        coods[i]['width']=coods[i]['width']*WIDTH
        coods[i]['height']=coods[i]['height']*WIDTH
    return coods


#==================== function: makeRequests ====================#  
# defining a function to get bounding box coordinates and tags
# makes calls to other functions defined above 
def makeRequests(byteImages,threshold):
    tagsData = []
    bound_box_cood = []
    responseJSON =  []

    # futures = [...]
    loop = asyncio.get_event_loop()
    check = loop.run_until_complete(fetch_all(byteImages))
    # check = asyncio.run()
    # check = await fetch_all(byteImages)
    for i,response in enumerate(check):
        df = getDFfromResponse(response, threshold)
        listBx = boundingBoxListFromDF(df)
        listBx = fitAccordingToSize(listBx)
        taglist = df['tagName'].tolist()
        tagsData.append(taglist)
        bound_box_cood.append(listBx)
        responseJSON.append(df)


    return bound_box_cood,tagsData, responseJSON



# defining a function to check compliance or non-compliance for a frame
def isCompliant(tags):
    return (len(list(filter(lambda x: x == 'mask', tags))) >= len(list(filter(lambda x: x == 'person',tags)))) 


# Print for Compliance Label
def PrintCompliance(frames,i,font,lineType):
    frames[i] = cv2.rectangle(
        frames[i], (WIDTH - 300, 0), (WIDTH, 50), (0, 128, 0), -1)
    frames[i] = cv2.putText(frames[i], "COMPLIANT",
                            (WIDTH - 220, 33),
                            font,
                            1,
                            (255, 255, 255), 2,
                            lineType)

# Print for Non-Compliance Label
def PrintNonCompliance(frames, i, font, lineType):
    frames[i] = cv2.rectangle(
        frames[i], (WIDTH - 300, 0), (WIDTH, 50), (0, 0, 128), -1)
    frames[i] = cv2.putText(frames[i], "NON-COMPLIANT",
                            (WIDTH - 275, 34),
                            font,
                            1,
                            (255, 255, 255), 2,
                            lineType)  

#==================== function: printCNC ====================#  
#Check for compliance or non-compliance label
def printCNC(which, frames, i, font, lineType):
    if which:
        PrintCompliance(frames, i, font, lineType)
    else:
        PrintNonCompliance(frames, i, font, lineType)   



# defining a function to convert a list of frames into a video 
def SaveVideo(frames, output_file_name):
    img_array = []
    for img in frames:
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(output_file_name,cv2.VideoWriter_fourcc(*'MP4V'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()                                



# defining a function to convert raw input video to processed video with tags and stats
def ConvertVideo(video_path, output_file_name, threshold, nframes):
    frames = getFrames(video_path) # this converts the input video into frames   
    
    if nframes != 'all':
        frames = frames[:int(nframes)]
    
    byteImages = convertCVImagesToByteImages(frames) # this converts frames into list of byte images
    
    coodsData,tagsData, responses = makeRequests(byteImages, threshold) # this returns lists of coordinate data, tag data, responses in dataframe   
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50,50)
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 1
    frames_stats = []

    for i,coods_single in enumerate(coodsData):
        dflist = responses[i].drop('boundingBox', axis=1)
        print(dflist['tagId'])
        dflist['tagId'] = (dflist['tagId'].to_string())[-10:]
        dflist.reset_index(drop=True, inplace=True)
        d = dflist.to_dict(orient='records')
        jo = json.dumps(d)
        width = 600
        height = 1080
        blank_image=np.zeros((height, width, 3), np.uint8)
        n_masks = len([x for x in tagsData[i] if x == 'mask'])
        n_people = len([x for x in tagsData[i] if x == 'person'])
        color = (0, 0, 0)
        blank_image[:height//2] = color
        blank_image[height//2:] = (0, 0, 0)
        blank_image = cv2.putText(blank_image, "STATISTICS" ,(230, 70),font, 1,(66, 245,173),2 ,lineType)
        blank_image = cv2.rectangle(blank_image, (80,150), (width-60, 220), (66, 245,173), 2)
        blank_image = cv2.rectangle(blank_image, (80, 260), (width-60, 330), (66, 245, 173), 2)
        blank_image = cv2.putText(blank_image, "NO OF PEOPLE:         "+ str(n_people),(90+20, 195),font, 0.9,(66, 245,173),2 ,lineType)
        blank_image = cv2.putText(blank_image, "NO OF MASKS:          "+ str(n_masks),(90+20, 305),font, 0.9,(66, 245,173),2 ,lineType)
        relative_start = height//2 + - 150
        x, y0 = (50, relative_start)
        text = dflist.to_string()
        thickness = 10
        text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
        line_height = text_size[1] + 20
    
        if len(text) != 0:
            for k, line in enumerate(text.split("\n")):
                y = y0 + k * line_height
                relative_start = y
                if k == 0:
                    ctext = '      Probability      Id          Tag'
                    blank_image = cv2.putText(blank_image, ctext, (x, y), font, 0.8, (66, 245,173),2, lineType)
                else:
                    blank_image = cv2.putText(blank_image, line, (x, y), font, 0.8, (66, 245,173),2, lineType)
   
        wrapped_text = textwrap.wrap(jo, width=50)
        x, y0 = (10, relative_start)
    
        if relative_start > 700:
            y = relative_start
        else:
            y = 700
    
        for p, line in enumerate(wrapped_text):
            print(line)
            y = y + 30
            blank_image =  cv2.putText(blank_image, line, (10, y), font, 0.7, (66, 245,173),2, lineType)

        frames_stats.append(blank_image)
        last = None

        if len(tagsData[i]) == 0:
            pass

        elif not isCompliant(tagsData[i]): # this checks for compliance for the frame
            if i % 5 == 0:
                PrintNonCompliance(frames, i, font, lineType) # this prints non-compliant labels   
                last = 0
            else:
                if last == None:
                    PrintNonCompliance(frames, i, font, lineType) 
                    last = 0
                else:
                    printCNC(last, frames, font, lineType) # this checks for compliance before applying compliance labels

        else:
            if i % 5 == 0:
                PrintCompliance(frames, i, font, lineType) # this prints compliant labels 
                last = 1
            else:
                if last == None:
                    PrintCompliance(frames, i, font, lineType)
                    last = 1
                else:
                    printCNC(last, frames, font, lineType)

        for j,single in enumerate(coods_single):
            # this puts the tab on top of the detected item
            frames[i] = cv2.rectangle(frames[i],  (int(single['left']) - 3, int(single['top']) - 30), ((int(single['left']) + 100, int(single['top']) - 3)), (66, 245,173), -1)
            frames[i] = cv2.putText(frames[i],tagsData[i][j],
                (int(single['left'] + 2), int(single['top']) + - 8),
                font,
                0.9,
                (0,0,0),2,
                lineType)

            # this creates the bounding box
            frames[i] = cv2.rectangle(frames[i],  (int(single['left']), int(single['top'])), (int(single['left'])+int(single['width']), int(single['top'])+int(single['height'])), (66, 245,173) ,3)
       
        frames[i] = np.concatenate((frames[i], frames_stats[i]), axis=1)

    SaveVideo(frames, output_file_name) # this converts frames into a video
    stats_path = output_file_name.replace('Output', 'Stats')
    SaveVideo(frames_stats, stats_path)                   