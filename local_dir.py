import os

test_id = '5'
data = "/home/pkrush/find-parts-faster-data/" + test_id + "/"
yes = "/home/pkrush/find-parts-faster-data/" + test_id + "/train/yes/"
no = "/home/pkrush/find-parts-faster-data/" + test_id + "/train/no/"
model = "/home/pkrush/find-parts-faster-data/" + test_id + "/model/"
warped = "/home/pkrush/find-parts-faster-data/" + test_id + "/warped/"
train = "/home/pkrush/find-parts-faster-data/" + test_id + "/train/"
capture = "/home/pkrush/find-parts-faster-data/" + test_id + "/capture/"


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def init_directories():
    dirs = [data,yes,no,model,warped]
    for new_dir in dirs:
        make_dir(new_dir)
