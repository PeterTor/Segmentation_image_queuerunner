import glob, os

class cityscapes_filereader:

    def __init__(self,path_to_images=None,path_to_gt=None,kind_of_training="instanceTrainIds"):

        if path_to_images:
            self.image_path = path_to_images
        else:
            self.image_path = "/media/peters/Data/cityscapes/leftImg8bit/*/*/"

        if path_to_gt:
            self.gt_path = path_to_gt
        else:
            self.gt_path ="/media/peters/Data/cityscapes/gtFine/*/*/"

        self.kind = kind_of_training


    #@outputs pais of images [image,labelimage]
    def get_image_pairs(self):

        gt = glob.glob(os.path.join(self.gt_path, '*.png'))
        imgs = glob.glob(os.path.join(self.image_path, '*.png'))
        result = []

        for i in imgs:
            filename = os.path.basename(i)  # file to copy
            filename_wo_png = filename.split(".")[0].replace("_leftImg8bit","")
            matching_path = [s for s in gt if filename_wo_png in s and "instanceTrainIds" in s]  # the correct path
            try :
                found_gt = matching_path[0]
            except:
                print "could findt gt for image",filename
            result.append([i, found_gt])
        return result


