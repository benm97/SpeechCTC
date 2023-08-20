import os

DATA_BASE_PATH = "dataset/an4"
LABEL_DIR = "txt"
DATA_DIR = "wav"


def load_data(data_type: str = "train"):
    data = {}
    data_path = os.path.join(DATA_BASE_PATH, data_type)
    i = 0
    for file in os.listdir(os.path.join(data_path, DATA_DIR)):
        if not file.endswith(".wav"):
            continue
        # extract label
        label_path = os.path.join(data_path, LABEL_DIR, os.path.splitext(
                file)[0] + ".txt")
        with open(label_path, "r") as label:
            data[i] = {'label': label.read(), 'audio_path': os.path.join(data_path,
                                                                        DATA_DIR, file)}
        i += 1
    return data


if __name__ == '__main__':
    data = load_data(data_type='train')
    print("SUCCESS")