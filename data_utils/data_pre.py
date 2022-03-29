import cv2
import numpy as np
import os
import random
from random import sample


class data_pre():
    """
    This is used as the preparion of the medical DSA images
    """

    def __init__(self):
        self.file_name = []  # for the func self.folder_index, a list plays the role of a stack in the func

    def image_converse(self, path):
        """
        This is the func for some images that have been conversed in the intensity of pixels.
        func will make the converse and save the processed image
        param path: the path of the image, of the name of the image
        return:
        """
        f = cv2.imread(path)
        f = 255 - f  # make the converse
        cv2.imwrite(path, f,
                    [cv2.IMWRITE_JPEG_QUALITY,
                     100])  # jpg images should be saved in the 100 mode in opencv, or it will lose some information

    def image_resize(self, path):
        """
        Resize the image to the size of 512*512
        param path: the image file path
        return: save the resized image
        """
        f = cv2.imread(path, flags=2)
        f = cv2.resize(f, (512, 512), interpolation=cv2.INTER_LINEAR)  # resize the image
        cv2.imwrite(path, f,
                    [cv2.IMWRITE_JPEG_QUALITY,
                     100])  # jpg images should be saved in the 100 mode in opencv, or it will lose some information

    def folder_rename(self, folder_path, new_name):
        """
        This is the func for the rename of the whole folder
        param path: the path of the folder
            new_name: new name of the files
        return: all the files in the folder will be renamed
        """
        folder_list = os.listdir(folder_path)
        for name in folder_list:
            old_name = folder_path + "\\" + name
            _name = folder_path + "\\" + new_name + name
            os.rename(old_name, _name)
        print("successfully renamed %d files" % (len(folder_list)))

    def folder_converse(self, folder_path):
        """
        Converse the whole images in a folder.
        param folder_path: the path of the folder
        return: converse all the images in the folder, and save the processed images in the folder
        """
        folder_list = os.listdir(folder_path)
        for image_name in folder_list:
            self.image_converse(folder_path + '\\' + image_name)
        print("%d images processed" % (len(folder_list)))

    def rand_choose(self, folder_path, choice_path, fraction):
        """
        Some images are almost the same, we random choose some part of them as the dataset
        para folder_path(str): the path of the folder
            fraction(float, <1): the proportion of the images we want to choose
            choice_path(str): the new folder to save the chosen images, if not exit, it will make a new one
        return: the chosen images will be saved in a new folder
        """
        folder_list = os.listdir(folder_path)
        random.seed(42)
        image_choice = random.sample(folder_list, int(fraction * len(folder_list)))  # choose the images randomly

        if not os.path.exists(choice_path):
            os.makedirs(choice_path)  # create the folder for the chosen files
        for image_name in image_choice:
            f = cv2.imread(folder_path + "//" + image_name, flags=2)
            if not f.shape == (512, 512):
                f = cv2.resize(f, (512, 512), interpolation=cv2.INTER_LINEAR)  # resize the image
            cv2.imwrite(choice_path + "//" + image_name, f,
                        [cv2.IMWRITE_JPEG_QUALITY,
                         100])  # jpg images should be saved in the 100 mode in opencv, or it will lose some information
        print("successfully processed %d images" % (len(image_choice)))

    def folder_index(self, folder_path):
        """
        This is the function to number the images in the whole dataset
        it will be performed in the way of recursion algorithm
        folder_path(str): the whole folder path of the data
        return: all the images will be renamed and resized
        """
        file_list = os.listdir(folder_path)
        if len(file_list) == 0:
            self.file_name.pop()  # if it is empty, make the pop of the stack, the empty folder won't make the for loop in the next lines

        for i in range(len(file_list)):
            if os.path.isdir(folder_path + '\\' + file_list[i]):
                self.file_name.append((file_list[i].split('_'))[0])  # self.filename is like the global variate
                self.folder_index(folder_path + '\\' + file_list[i])  # make the recursion
            else:
                old_name = folder_path + '\\' + file_list[i]
                self.image_resize(old_name)  # resize the image
                new_name = folder_path + '\\' + (self.file_name)[0] + '_' + (self.file_name)[1] + '_' + (self.file_name)[2] + '_' + str(i) + '.jpg'
                # new_name = folder_path + '\\' + ''.join(self.file_name) + '_' + str(i) + '.jpg'
                os.rename(old_name, new_name)  # change the name

            if (i == (len(file_list) - 1)) and (len(self.file_name) != 0):  # if it is the last image in the folder, self.filename stack should make the pop
                self.file_name.pop()


test = data_pre()
# test.data_converse(path='IMG-0001-00001.jpg')  #  a test of the image
# test.image_resize('IMG-0001-00016.jpg')
folder_paths = ["H:\\QFR_dataset_2\\0_upper_limb\\upper_limb_PkUFH\\2_chenshanwen\\CHEN_SHAN_WEN_61\\white_converse",
                "H:\\QFR_dataset_2\\0_upper_limb\\upper_limb_PkUFH\\7_liuzhixiang\\LIU_ZHI_XIANG_65\\white_converse",
                "H:\\QFR_dataset_2\\0_upper_limb\\upper_limb_PkUFH\\8_fanbaoyang\\FAN_BAO_YANG_61\\white_converse",
                "H:\\QFR_dataset_2\\0_upper_limb\\upper_limb_PkUFH\\9_yangjingrong\\YANG_JING_RONG_76\\white_converse"
                ]
# test.folder_converse(folder_paths[0])
"""frac = 0.05
paths = folder_paths[0]
test.rand_choose(paths, paths + '_' + str(0.05), frac)"""
"""for paths in folder_paths:
    test.rand_choose(paths, paths + '_' + str(0.05), frac)"""
# path = r"H:\QFR_dataset_2\1_lower_limb\1_lower_limb_PUMCH\12_sunjinfang\sunjinfang-1\2017_12_4_11_25_49"
# test.folder_rename(path, "I")
folder_path = r"H:\QFR_dataset_7"
test.folder_index(folder_path)
