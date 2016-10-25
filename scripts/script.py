import glob
filelist = glob.glob('/home/arpit/caffe/caffe-master/data/img_rating/*.JPEG')
filelist_file = open(
    '/home/arpit/caffe/caffe-master/data/img_rating/list.txt', 'w')
for filename in filelist:
    filelist_file.write(filename)
    filelist_file.write('\n')
filelist_file.close()
