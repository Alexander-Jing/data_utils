import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from random import sample
import shutil
from skimage.measure import compare_psnr


class data_pre():
    """
    This is used as the preparion of the medical DSA images
    """

    def __init__(self):
        self.file_name = []  # for the func self.folder_index, a list plays the role of a stack in the func
        self.remove_count = 0  # for the func self.remove_similarity, count how many images deleted

    def image_converse(self, path):
        """
        This is the func for some images that have been conversed in the intensity of pixels.
        func will make the converse and save the processed image
        param path: the path of the image, of the name of the image
        return:
        """
        f = cv2.imread(path, flags=2)
        f = 255 - f  # make the converse
        cv2.imwrite(path, f,
                    [cv2.IMWRITE_JPEG_QUALITY,
                     100])  # jpg images should be saved in the 100 mode in opencv, or it will lose some information

    def image_resize(self, path, size):
        """
        Resize the image to the size of 512*512
        param path: the image file path
        return: save the resized image
        """
        f = cv2.imread(path, flags=2)
        f = cv2.resize(f, size, interpolation=cv2.INTER_LINEAR)  # resize the image
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
        folder_path(str): the whole folder path of the data(root folder)
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
                self.image_resize(old_name, (512, 512))  # resize the image
                new_name = folder_path + '\\' + (self.file_name)[0] + '_' + (self.file_name)[1] + '_' + \
                           (self.file_name)[2] + '_' + str(i) + '.jpg'
                # new_name = folder_path + '\\' + ''.join(self.file_name) + '_' + str(i) + '.jpg'
                os.rename(old_name, new_name)  # change the name

            if (i == (len(file_list) - 1)) and (
                    len(self.file_name) != 0):  # if it is the last image in the folder, self.filename stack should make the pop
                self.file_name.pop()

    def batch_refine(self, path, imlist, per):
        """
        This is the func to find the best projection image in the series of DSA images
            choose the best image in the series(actually it is a video)
            we calculate the differential, in order to find the best fitted.
        :param path: the path of the sub folder
        :param imlist: list of the image names, take care the images in the list should be in the same series
        :param per: the percentage of the series of images want to be left
        :return: the best fitted projection DSA images will be left, others will be deleted
        """
        im_series = []  # list for the images
        for _name in imlist:
            im_series.append((cv2.imread(path + "//" + _name, flags=2)).astype(np.float16) / 255.0)
        im_series = np.array(im_series)  # read the images as a series list
        _diff = np.diff(im_series, axis=0)
        _diff = np.sum(np.abs(_diff), axis=(1, 2))  # calculate the differential of the series
        plt.plot(np.array([i + 1 for i in range(_diff.shape[0])]), _diff)
        plt.show()
        im_in = np.argmin(_diff)  # locate the smallest differential, which is the best projection image
        _list = imlist[int(im_in - (1 / 2) * per * len(imlist)): int(
            im_in + (1 / 2) * per * len(imlist))]  # find some sub optimal near the best
        for _names in _list:
            self.image_resize(path + "//" + _names, (512, 512))  # resize and save

        del imlist[int(im_in - per * len(imlist)): int(im_in + per * len(imlist))]
        for _renames in im_list:
            os.remove(path + "//" + _renames)

    def batch_refine_pro(self, path, imlist, per):
        """
        This is the func to find the best projection image in the series of DSA images
            choose the best image in the series(actually it is a video), but we compare the images with the first image
            so that we can find the best fitted
        :param path: the path of the sub folder
        :param imlist: list of the image names, take care the images in the list should be in the same series
        :param per: the percentage of the series of images want to be left
        :return: the best fitted projection DSA images will be left, others will be deleted
        """
        im_series = []  # list for the images
        for _name in imlist:
            im_series.append((cv2.imread(path + "//" + _name, flags=2)).astype(np.float16) / 255.0)
        im_series = np.array(im_series)  # read the images as a series list
        _diff = im_series - im_series[0]
        _diff = np.sum(np.abs(_diff), axis=(1, 2))  # calculate the differential of the series
        # plt.plot(np.array([i + 1 for i in range(_diff.shape[0])]), _diff)
        # plt.show()
        im_in = np.argmax(_diff)  # locate the largest one, which is the best projection image
        _list = imlist[int(im_in - (1 / 2) * per * len(imlist)) + 1: int(
            im_in + (1 / 2) * per * len(imlist)) + 1]  # find some sub optimal near the best
        for _names in _list:
            self.image_resize(path + "//" + _names, (512, 512))  # resize and save

        del imlist[int(im_in - (1 / 2) * per * len(imlist)) + 1: int(im_in + (1 / 2) * per * len(imlist)) + 1]
        for _renames in imlist:
            os.remove(path + "//" + _renames)

    def sub_folder_refine(self, path, per):
        """
        This is the func based on the batch_refine_pro func in order to deal with the series of DSA images in the folder
        the difficulty is to deal with the name of the files, and try to local the series via the file names
        :param path: the path of the sub folder
        :param per: the percentage of the series of images want to be left
        :return: the best fitted projection DSA images will be left, others will be deleted
        """
        file_list = os.listdir(path)

        iim_lists = []
        num = 0
        while (True):
            num += 1
            if num < 10:
                _name = 'IIMG-' + '000' + str(num)
                c = [i for i in file_list if _name in i]
            else:
                _name = 'IIMG-' + '00' + str(num)
                c = [i for i in file_list if _name in i]
            if len(c) == 0:
                break
            else:
                iim_lists.append(c)  # get the list of names of the 'IIMG' images

        for _files in iim_lists:
            for _file in _files:
                file_list.remove(_file)  # delete the 'IIMG' files

        im_lists = []
        num = 0
        while (True):
            num += 1
            if num < 10:
                _name = 'IMG-' + '000' + str(num)
                c = [i for i in file_list if _name in i]
            else:
                _name = 'IMG-' + '00' + str(num)
                c = [i for i in file_list if _name in i]
            if len(c) == 0:
                break
            else:
                im_lists.append(c)  # get the list of names of the 'IMG' images

        for _list in im_lists:
            if len(_list) != 0:
                self.batch_refine_pro(path, _list, per)
        if len(iim_lists) != 0:
            for _list in iim_lists:
                if len(_list) != 0:
                    self.batch_refine_pro(path, _list, per)  # make the refine

    def folder_refine(self, folder_path, per):
        """
        This is the func based on the sub_folder_refine func in order to deal with the series of DSA images in the folder
        the difficulty is to deal with the name of the files, and try to local the series via the file names
        :param path: the path of the root folder
        :param per: the percentage of the series of images want to be left
        :return: the best fitted projection DSA images will be left, others will be deleted
        """
        file_list = os.listdir(folder_path)
        if len(file_list) == 0:
            pass
        for _file in file_list:
            if os.path.isdir(folder_path + '\\' + _file):
                self.folder_refine(folder_path + '\\' + _file, per)  # make the recursion
            else:
                self.sub_folder_refine(folder_path, per)  # the func can be modified if needed
                break  # write the break, or the sub_folder_refine will be operated more times

    def image_compare(self, img1, img2):
        """
        This is the func that compares the two images im1,im2
        :param img1, img2: the two compared images
        :return: if the same, return True, else False
        """
        img2 = cv2.resize(img2, img1.shape, interpolation=cv2.INTER_LINEAR)
        psnr = compare_psnr(img1, img2)
        if psnr >= 30:  # psnr >30dB, two images are almost the same
            return True
        else:
            return False

    def image_compare_pro(self, img1, img2, threshold):
        """
        This is the func that compares the two images im1,im2
        :param img1, img2: the two compared images
        :param threshold: the threshold of judging the similarity of two images(larger than 30)
        :return: if the same, return True, else False
        """
        img2 = cv2.resize(img2, img1.shape, interpolation=cv2.INTER_LINEAR)
        psnr = compare_psnr(img1, img2)
        if psnr >= threshold:  # psnr >30dB, two images are almost the same
            return True
        else:
            return False

    def folder_compare(self, img, path):
        """
        Compare the images in a folder with the image
        :param path: folder path
        :param img: image for comparing
        :return: if found the same image in the folder, return true, else return false
        """
        file_list = os.listdir(path)
        if len(file_list) != 0:
            for _file in file_list:
                im1 = cv2.imread(path + "//" + _file, flags=2)
                _result = self.image_compare(img, im1)  # compare the image
                if _result:
                    return True
        return False

    def folders_compare(self, folder1, folder2):
        """
        Compare the images in two folders
        :param folder1: the images in folder1
        :param folder2: the images in folder2
        :return: if the same, delete the same image in folder1
        """
        folder_list1 = os.listdir(folder1)
        folder_list2 = os.listdir(folder2)
        for i in range(len(folder_list1)):
            _list = os.listdir(folder1 + "\\" + folder_list1[i])
            for _patientName in _list:
                num = 0
                _imgs = os.listdir(folder1 + '\\' + folder_list1[i] + '\\' + _patientName)
                for _img in _imgs:
                    img = cv2.imread(folder1 + '\\' + folder_list1[i] + '\\' + _patientName + "\\" + _img, flags=2)
                    if os.path.exists(folder2 + '\\' + folder_list2[i] + '\\' + _patientName):
                        _result = self.folder_compare(img, folder2 + '\\' + folder_list2[i] + '\\' + _patientName)
                        if _result:
                            os.remove(folder1 + '\\' + folder_list1[i] + '\\' + _patientName + "\\" + _img)
                            num += 1

                print(_patientName + " " + "removed %d images" % (num))

    def rename_folder(self, folder_path, dis_path):
        """make the rename and move the images"""
        file_list = os.listdir(folder_path)
        if len(file_list) == 0:
            self.file_name.pop()  # if it is empty, make the pop of the stack, the empty folder won't make the for loop in the next lines

        for i in range(len(file_list)):
            if os.path.isdir(folder_path + '\\' + file_list[i]):
                self.file_name.append((file_list[i].split('_'))[0])  # self.filename is like the global variate
                self.rename_folder(folder_path + '\\' + file_list[i], dis_path)  # make the recursion
            else:
                old_name = folder_path + '\\' + file_list[i]
                self.image_resize(old_name, (512, 512))  # resize the image
                new_name = folder_path + '\\' + (self.file_name)[0] + '_' + (self.file_name)[1] + '_' + \
                           (self.file_name)[2] + '_' + str(i) + '.jpg'
                # new_name = folder_path + '\\' + ''.join(self.file_name) + '_' + str(i) + '.jpg'
                os.rename(old_name, new_name)  # change the name
                # make the movement
                if self.file_name[1] == '0':
                    shutil.copy(new_name, os.path.join(dis_path, (self.file_name)[2]))
                if self.file_name[1] == '1':
                    shutil.copy(new_name, os.path.join(dis_path, str(10 + int((self.file_name)[2]))))
                if self.file_name[1] == '2':
                    shutil.copy(new_name, os.path.join(dis_path, str(24 + int((self.file_name)[2]))))
            if (i == (len(file_list) - 1)) and (
                    len(self.file_name) != 0):  # if it is the last image in the folder, self.filename stack should make the pop
                self.file_name.pop()

    def rename_folder_pro(self, folder_path):
        """
        rename the names of images in the new folder
        :param folder_path: the root folder path
        :return: names of images in the folder will be changed
        """
        folder_list = os.listdir(folder_path)
        for i in range(len(folder_list)):
            file_list = os.listdir(os.path.join(folder_path, folder_list[i]))
            for j in range(len(file_list)):
                old_name = os.path.join(folder_path, folder_list[i], file_list[j])
                new_name = os.path.join(folder_path, folder_list[i],
                                        folder_list[i].zfill(3) + '_' + str(j).zfill(4) + '.jpg')
                os.rename(old_name, new_name)
            print("patient " + folder_list[i].zfill(3) + " renamed %d images" % (len(file_list)))

    def remove_similarity(self, folder_path, threshold):
        """
        remove similar images in the folder, compare the image with the left images, if the same, delete the image
        :param folder_path: the path of the root folder, in the form of recursion
        :param threshold: the threshold of judging the similarity of two images(larger than 30)
        :return: delete the same images
        """
        file_list = os.listdir(folder_path)
        if len(file_list) == 0:
            pass
        for i in range(len(file_list)):
            if os.path.isdir(folder_path + '\\' + file_list[i]):
                self.remove_similarity(folder_path + '\\' + file_list[i], threshold)  # make the recursion
            else:
                old_name = folder_path + '\\' + file_list[i]
                im1 = cv2.imread(old_name, flags=2)
                if len(file_list[(i + 1):]) != 0:  # compare the left images
                    for _im in file_list[(i + 1):]:
                        im2 = cv2.imread(folder_path + '\\' + _im, flags=2)
                        if self.image_compare_pro(im1, im2, threshold):  # if the same
                            os.remove(old_name)
                            self.remove_count += 1
                            print(str(self.remove_count) + " remove image: " + old_name)
                            break


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
folder_path = r"H:\QFR_dataset_popliteal"
path = r'F:\CASIA\codes\data_utils\4_lilianwei_1'

im_list = ['IMG-0009-00001.jpg',
           'IMG-0009-00002.jpg',
           'IMG-0009-00003.jpg',
           'IMG-0009-00004.jpg',
           'IMG-0009-00005.jpg',
           'IMG-0009-00006.jpg',
           'IMG-0009-00007.jpg',
           'IMG-0009-00008.jpg',
           'IMG-0009-00009.jpg',
           'IMG-0009-00010.jpg',
           'IMG-0009-00011.jpg',
           'IMG-0009-00012.jpg',
           'IMG-0009-00013.jpg',
           'IMG-0009-00014.jpg',
           'IMG-0009-00015.jpg',
           'IMG-0009-00016.jpg']
# test.batch_refine_pro(path, im_list, 0.2)
# test.folder_index(folder_path)
# test.sub_folder_refine(path, 0.1)
# test.folder_refine(folder_path, 0.1)
# test.folders_compare(r"H:\QFR_dataset_v1.5\2_popliteal", r"H:\QFR_dataset_v1.5\0_upper_knee")
# test.rename_folder(r"H:\QFR_dataset_v1.5", r"H:\QFR_dataset_v2.0")
test.rename_folder_pro("H:\QFR_dataset_v5.5")
# test.remove_similarity("H:\QFR_dataset_v5.0", 45)
