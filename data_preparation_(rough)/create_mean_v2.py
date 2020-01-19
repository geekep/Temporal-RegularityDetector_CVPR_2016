import os
import PIL
import numpy as np
from PIL import Image

# input paths for avenue
# training
# frame_dir = 'C:/Users/admin/Downloads/Surveillance/Avenue Dataset/training_frames_gray'
# flow_dir = 'C:/Users/admin/Downloads/Surveillance/Avenue Dataset/training_flow'
# save_path = 'C:/Users/admin/Downloads/Surveillance/Avenue Dataset/training_hdf5'
# num_videos = 16

# input paths for ucsd ped1
# training
# frame_dir = 'C:/Users/admin/Downloads/Surveillance/ped1/training_frames_gray'
# flow_dir = 'C:/Users/admin/Downloads/Surveillance/ped1/training_flow'
# save_path = 'C:/Users/admin/Downloads/Surveillance/ped1/training_hdf5'
# num_videos = 34

# input paths for ucsd ped2
# training
# frame_dir = 'C:/Users/admin/Downloads/Surveillance/ped2/training_frames_gray'
# flow_dir = 'C:/Users/admin/Downloads/Surveillance/ped2/training_flow'
# save_path = 'C:/Users/admin/Downloads/Surveillance/ped2/training_hdf5'
# num_videos = 16

# input paths for subway enter
# training
# frame_dir = 'C:/Users/admin/Downloads/Surveillance/Amit-Subway/enter/training_frames_gray'
# flow_dir = 'C:/Users/admin/Downloads/Surveillance/Amit-Subway/enter/training_flow'
# save_path = 'C:/Users/admin/Downloads/Surveillance/Amit-Subway/enter/training_hdf5'
# num_videos = 1

# input paths for subway exit
# training
frame_dir = 'C:/Users/admin/Downloads/Surveillance/Amit-Subway/exit/training_frames_gray'
flow_dir = 'C:/Users/admin/Downloads/Surveillance/Amit-Subway/exit/training_flow'
save_path = 'C:/Users/admin/Downloads/Surveillance/Amit-Subway/exit/training_hdf5'
num_videos = 1

num_row = 227
num_col = 227

# allocate memory
data_only_frames = np.zeros((num_row, num_col)).astype('float64')
data_frames_flows = np.zeros((num_row, num_col)).astype('float64')

total_count = 0
for i in range(0, num_videos):
	video_name = '%02d' % (i+1)
	print('==> '+video_name)

	video_frame_folder = os.path.join(frame_dir, video_name)
	frames_path = [f for f in os.listdir(video_frame_folder)]
	frames_path.sort()

	video_flow_folder = os.path.join(flow_dir, video_name)
	flows_path = [f for f in os.listdir(video_flow_folder)]
	flows_path.sort()
	num_frames = len(frames_path)

	# read and process all the frames and flows for this video from the disk
	for j in range(0, num_frames, 5):
		frame_path = os.path.join(frame_dir, video_name, frames_path[j])
		frame = Image.open(frame_path)
		frame = frame.resize((num_row, num_col), PIL.Image.ANTIALIAS)
		frame = np.array(frame, order='C').astype('float16')
		frame = frame / 255
		data_only_frames = data_only_frames + frame

		flow_path = os.path.join(flow_dir, video_name, flows_path[j])
		flow = Image.open(flow_path)
		flow = flow.resize((num_row, num_col), PIL.Image.ANTIALIAS)
		flow = np.array(flow, order='C').astype('float16')
		flow = flow / 255
		data_frames_flows = data_frames_flows + flow
		
		total_count = total_count + 1

		if total_count == 10000:
			break

data_only_frames = data_only_frames / total_count
data_frames_flows = data_frames_flows / total_count

print('==> total '+str(total_count)+' instances')
print('==> saving mean')
np.save(os.path.join(save_path, 'mean_frame_227.npy'), data_only_frames)
np.save(os.path.join(save_path, 'mean_flow_227.npy'), data_frames_flows)

