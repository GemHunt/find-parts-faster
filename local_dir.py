import os
data = "/home/pkrush/find-parts-faster-data/3/"
yes = "/home/pkrush/find-parts-faster-data/3/train/yes/"
no =  "/home/pkrush/find-parts-faster-data/3/train/no/"
model = "/home/pkrush/find-parts-faster-data/3_10x/model/"
warped = "/home/pkrush/find-parts-faster-data/3/warped/"


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def init_directories():
    dirs = [data,yes,no,model,warped]
    for new_dir in dirs:
        make_dir(new_dir)
