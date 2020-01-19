../all_datasets_10frames_3seqs contains input hdf5 files for all five datasets - avenue, ped1, ped2, enter, and exit
For each dataset and each video, input cubes are 10 frames in temporal length. To generate this 10 frames, we used following three types of sequences - 
0 1 2 3 4 5 6 7 8 9
0 2 4 6 8 10 12 14 16 18
0 3 6 9 12 15 18 21 24 27

The network architecture is defined in train_conv3.prototxt file. It is - [10 256 128 64 128 256 10]
