from zipfile import ZipFile
import os
from os.path import basename


def zip_files_in_dir(dir_name, zip_file_name, filter):
    with ZipFile(zip_file_name, 'w') as zipObj:
        for folder_name, subfolders, filenames in os.walk(dir_name):
            for filename in filenames:
                if filter(filename):
                    filepath = os.path.join(folder_name, filename)
                    zipObj.write(filepath, basename(filepath))
