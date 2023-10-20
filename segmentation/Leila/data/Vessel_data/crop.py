import os
from PIL import Image
from scipy import misc

def crop_img(data_dir):

    images = [im for im in os.listdir(data_dir)]
    d_num = len(images)
    test = misc.imread(data_dir+images[0])
    im_size = test.shape
    z = 0
    while (z)<d_num:
        cr_z = './z' + str(z) + '-' + str(z+128)+ '/'
        if not os.path.exists(cr_z):
            os.mkdir(cr_z)
        for image in images[z:z+128]:
            img = Image.open(data_dir + image)
            h=0
            while (h)<im_size[0]:
                w=0
                while (w)< im_size[1]:
                    im_dir = cr_z + 'h' + str(h) + '-' + str(h+128)+ '_w' + str(w) + '-' + str(w+128)+ '/'
                    if not os.path.exists(im_dir):
                        os.mkdir(im_dir)
                    crop = img.crop((h, w, h+128, w+128))
                    name = im_dir + image
                    crop.save(name)

                    w = w + 118


                h = h+118

        z=z+118



data_dir='./Test_data/'

crop_img(data_dir)
