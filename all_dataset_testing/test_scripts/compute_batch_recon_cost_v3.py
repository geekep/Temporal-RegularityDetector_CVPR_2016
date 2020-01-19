import os
import sys
import h5py

sys.path.insert(0, '/home/mhasan/caffe/python')
import caffe
import numpy as np
import progressbar

test_dir = sys.argv[1]
deploy_prototxt = sys.argv[2]
caffe_model = sys.argv[3]
batch_size = int(sys.argv[4])
save_path = sys.argv[5]
first_video = int(sys.argv[6])
num_videos = int(sys.argv[7])
file_name_prefix = sys.argv[8]
stride = int(sys.argv[9])
wchar = sys.argv[10]

if not os.path.exists(save_path):
    os.makedirs(save_path)

net = caffe.Net(deploy_prototxt, caffe_model, caffe.TEST)

for i in range(first_video - 1, num_videos + first_video - 1):
    vid_name = file_name_prefix + '_' + '%02d' % (
            i + 1) + '_frames_10_stride_' + '%0d' % stride + '_batch_' + '%02d' % (i + 1)
    vid_dir = os.path.join(test_dir, vid_name + '.h5')
    print(vid_dir)

    f = h5py.File(vid_dir, 'r')
    data = f['data']
    num_input = data.shape[0]
    num_channel = data.shape[1]
    costs = np.zeros(num_input, dtype=float)

    bar = progressbar.ProgressBar(maxval=num_input,
                                  widgets=[progressbar.Bar('>', '[', ']'), ' ', progressbar.SimpleProgress(), ' ',
                                           progressbar.Percentage(), ' ', progressbar.ETA()]).start()

    count = 0
    for j in range(0, num_input, batch_size):
        bar.update(j)

        if j + batch_size > num_input:
            break

        net.blobs['data'].data[...] = data[j:j + batch_size]
        net.forward()
        fout = net.blobs['deconv1'].data
        for k in range(batch_size):
            costs[j + k] = np.linalg.norm(np.squeeze(fout[k]) - np.squeeze(data[j + k]))
            count = count + num_channel

    bar.finish()
    np.savetxt(os.path.join(save_path, file_name_prefix + '_' + '%02d' % (i + 1) + '_' + wchar + '.txt'), costs)

'''bash
python test_scripts/compute_batch_recon_cost_v3.py
       ./all_dataset_10frames_stride5 [test_dir]
       ./prototxts/deploy_conv3.prototxt [deploy_prototxt]
       ./trained_models/conv_auto_all_iter_150000.caffemodel [caffe_model]
       1 [batch_size]
       ./recon_costs [save_path]
       21 [num_videos]
       avenue_video [file_name_prefix]
       5 [stride]
       conv3_iter_10000 [wchar]
'''