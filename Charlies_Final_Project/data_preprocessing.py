from PIL import Image
import os

import function_sandbox as helper

def clean_data(path_to_raw_dir='/Users/johnrogermepham/Documents/University of San Diego/Senior/Fall 2018/comp_380/'
               +'Final_Project/Data/Dogs_Cats/Raw_Data', path_to_clean_dir = '/Users/johnrogermepham/Documents/'
               + 'University of San Diego/Senior/Fall 2018/comp_380/Final_Project/Data/Dogs_Cats/Clean_Data'):
    classes = helper.get_sub_dirs(path_to_raw_dir)

    for individual_class in classes:
        image_paths =  helper.get_sub_dirs(individual_class)
        class_name =  os.path.basename(os.path.normpath(individual_class))
        path = os.path.join(path_to_clean_dir, class_name)
        count = 0
        for individual_image in image_paths:
            try:
                temp = Image.open(individual_image)
            except IOError:
                print("FAULURE TO OPEN IMAGE")
                continue

            temp = temp.resize((224, 224), Image.ANTIALIAS)
            flipped = temp.transpose(Image.FLIP_LEFT_RIGHT)

            try:
                temp.save(os.path.join(path, str(count) + '.jpg'))

                flipped.save(os.path.join(path, str(count+1) + '.jpg'))

            except IOError:
                print('FAILURE TO SAVE IMAGE')
                continue
            count += 2

