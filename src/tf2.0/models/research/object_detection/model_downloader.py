import tarfile
tar = tarfile.open('efficientdet_d0_coco17_tpu-32.tar.gz')
tar.extractall('.')
tar.close()