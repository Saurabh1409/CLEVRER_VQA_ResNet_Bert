import argparse
import numpy as np
import cv2
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',type = str)
    parser.add_argument('--output_path',type = str)
    parser.add_argument('--size',type = int)
    args = parser.parse_args() 
    
    input_folder = args.input_path
    output_folder = args.output_path
    video_extensions = [".mp4", ".avi"] 
    size = args.size
    frame_interval_ms = 200 

    for filename in os.listdir(input_folder):
        filepath1 = os.path.join(input_folder, filename)
        
        counter = 0
        for video in os.listdir(filepath1):
            filepath = os.path.join(filepath1, video)
            if os.path.splitext(video)[1] in video_extensions:
                cap = cv2.VideoCapture(filepath)
                frame_rate = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = int(frame_rate * frame_interval_ms / 1000) # compute the frame interval in frames
                frame_count = 0
                count_num = 0
                os.chdir(output_folder)
                os.mkdir(f"{os.path.splitext(video)[0]}")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame,(size,size))
                    if frame_count % frame_interval == 0:
                        frame_filename = f"{count_num}.jpg"
                        output_dir = output_folder+"/"+f"{os.path.splitext(video)[0]}"
                        frame_filepath = os.path.join(output_dir, frame_filename)
                        cv2.imwrite(frame_filepath, frame)
                        count_num += 1
                    frame_count += 1
                # Release the video file
                cap.release()
            print("Completed " + str(counter)+"!")
            counter+=1
    print("Complete!")



