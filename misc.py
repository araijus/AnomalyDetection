import os
import pickle


def dump_pickle(file_path, obj):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        obj = pickle.load(file)
    return obj


def build_files_list(root_dir):
    normal_files = []
    abnormal_files = []

    for root, _, files in os.walk(top=root_dir):
        for name in files:
            full_path = os.path.join(root, name)
            if root == "/.../ToyCar_data/NormalSound":
                normal_files.append(full_path)
            elif root == "/.../ToyCar_data/AnomalousSound":
                abnormal_files.append(full_path)

    return normal_files, abnormal_files