from torch.utils.data import Dataset

class DisparityDataset(Dataset):
    """
    Dataset class which provides items of the form (left image, right image), disparity map.
    
    It is based on cacheing the list of paths to left images in a list, and on the assumption that there is a systematic way to       convert the path to a left image into the paths to a right image and a corresponding disparity map. This conversion is assumed 
    to be described by a function left2right and left2disparity, respectively. 
    Since the images are often saved in different forms, and so are the disparity maps, a function which loads such an image or 
    disparity given the file path is required for the images and disparities. Transforms can also be applied, it should be of a 
    joint form:  
    (img_left_transformed, img_right_transformed), disparity_transformed <- transform((img_left, img_right), disparity)
    """ 
    
    def __init__(self, left_paths, left2right, left2disparity, load_image, load_disparity, transform=None):
        self.left_paths = left_paths
        self.right_paths = [left2right(path) for path in left_paths]
        self.disp_paths = [left2disparity(path) for path in left_paths]
        
        self.load_image = load_image
        self.load_disparity = load_disparity
        self.transform = transform
        return
    
    def __len__(self):
        """Return the number of file paths."""
        
        return len(self.left_paths)

    def __getitem__(self, i):
        """ 
        Loads left image, right image and disparity by corresponding file path at index i, finally apply transform.
        """
        
        left_img_path, right_img_path = self.left_paths[i], self.right_paths[i]
        disparity_path = self.disp_paths[i]
        
        left_img, right_img = self.load_image(left_img_path), self.load_image(right_img_path)
        disparity_map = self.load_disparity(disparity_path)

        if self.transform:
            (left_img, right_img), disparity_map = self.transform((left_img, right_img), disparity_map)

        return (left_img, right_img), disparity_map