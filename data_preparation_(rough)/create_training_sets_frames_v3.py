import os
import sys
import h5py
import PIL
import numpy as np
from PIL import Image
from collections import deque
import progressbar

frame_mean_path = sys.argv[1]
frame_dir = sys.argv[2]
save_path = sys.argv[3]
num_videos = int(sys.argv[4])
twin_len = int(sys.argv[5])
stride = int(sys.argv[6])
which_seq = int(sys.argv[7])

if not os.path.exists(save_path):
    os.makedirs(save_path)

batch_size = 2500
num_row = 227
num_col = 227
seqs = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
        [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]]
seq = seqs[which_seq]

# load means
mean_frame = np.load(frame_mean_path)

total_count = 0
batch_no = 1
for i in range(0, num_videos):
    video_name = '%02d' % (i + 1)
    print('==> ' + video_name)
    count = 0

    f1 = h5py.File(os.path.join(frame_dir, video_name + '.h5'), 'r')
    data_frames = f1['data']
    num_frames = data_frames.shape[0]

    # allocate memory
    data_only_frames = np.zeros((batch_size, len(seq), num_row, num_col)).astype('float16')

    # progress bar
    bar = progressbar.ProgressBar(maxval=num_frames,
                                  widgets=[progressbar.Bar('>', '[', ']'), ' ', progressbar.SimpleProgress(), ' ',
                                           progressbar.Percentage(), ' ', progressbar.ETA()]).start()

    # read and process all the frames and flows for this video from the disk
    deque_only_frames = deque([], maxlen=twin_len)
    for j in range(0, num_frames):
        bar.update(j + 1)
        frame = data_frames[j].astype('float16')
        frame = frame / 255 - mean_frame

        deque_only_frames.append(frame)

        if (j - twin_len) >= 0 and (j - twin_len) % stride == 0:
            kc = 0
            for k in seq:
                data_only_frames[count, kc] = deque_only_frames[k]
                kc = kc + 1
            count = count + 1

        if count == batch_size:
            print('\n==> writing ' + str(count) + ' instancess of ' + video_name + ' batch %02d' % batch_no)
            with h5py.File(os.path.join(save_path,
                                        'data_video_' + video_name + '_frames_' + str(twin_len) + '_stride_' + str(
                                            stride) + '_batch_%02d' % batch_no + '.h5'), 'w') as f:
                f['data'] = data_only_frames
                count = 0
                batch_no = batch_no + 1

    bar.finish()
    print('\n==> writing ' + str(count) + ' instancess of ' + video_name + ' batch %02d' % batch_no)
    with h5py.File(os.path.join(save_path,
                                'data_video_' + video_name + '_frames_' + str(twin_len) + '_stride_' + str(
                                    stride) + '_batch_%02d' % batch_no + '.h5'), 'w') as f:
        f['data'] = data_only_frames[0:count]
        count = 0
        batch_no = batch_no + 1

'''bash
python data_preparation_(rough)/create_training_sets_frames_v3.py 
       [frame_mean_path] 
       [frame_dir]
       [save_path]
       [num_videos]
       10 [twin_len]
       5 [stride]
       [which_seq]
'''
