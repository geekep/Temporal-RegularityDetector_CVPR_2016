import os
import numpy as np
import tensorflow as tf
import model
import cv2

AvenueDatasetPath = 'C:/Users/admin/Downloads/Surveillance/Avenue Dataset'
SubwayEnterPath = 'C:/Users/admin/Downloads/Surveillance/Amit-Subway/enter'
SubwayExitPath = 'C:/Users/admin/Downloads/Surveillance/Amit-Subway/exit'
UCSDped1Path = 'C:/Users/admin/Downloads/Surveillance/UCSD_Anomaly_Dataset.v1p2/UCSDped1'
UCSDped2Path = 'C:/Users/admin/Downloads/Surveillance/UCSD_Anomaly_Dataset.v1p2/UCSDped2'


def main():
    dirList = os.listdir(AvenueDatasetPath)
    videosPath = 'training_videos'

    if videosPath in dirList:
        os.path.join(AvenueDatasetPath, videosPath)
    else:
        print('{} not exist!'.format(videosPath))

    if os.path.isfile(videoAbsolutePath):

        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frameList = []
        while cap.isOpened():

            ret, frame = cap.read()
            if ret:

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (set_params.W, set_params.H), interpolation=cv2.INTER_CUBIC)
                print(gray.shape)
                frameList.append(gray)
                print('frame', ret, "of Video", v, "has been saved!")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    clip = np.array(clip)
    clip = np.float32(clip)

    with tf.Session() as sess:

        input_shape = (10, 227, 227, 3)
        trd = model.TemporalRegularityDetector(sess, input_shape)


if __name__ == '__main__':
    main()
