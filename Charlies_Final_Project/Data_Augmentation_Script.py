from PIL import Image, ImageFilter
import os
from pathlib import Path
import function_sandbox as helper


cats_train_dir = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Data\Dogs_Cats_Update\Partition_Data\Train\Cats')
dogs_train_dir = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Data\Dogs_Cats_Update\Partition_Data\Train\Dogs')

classes = [cats_train_dir, dogs_train_dir]
for individual_class in classes:
    base_name = str(os.path.basename(os.path.normpath(individual_class)))
    image_paths = helper.get_sub_dirs(individual_class)
    path = individual_class
    num_images = len(image_paths)
    count = 5000
    processed = 1
    for individual_image in image_paths:
        helper.progress(base_name, processed/num_images)
        processed += 1
        try:
            temp = Image.open(individual_image)
        except IOError:
            print("FAULURE TO OPEN IMAGE")
            continue

        temp = temp.resize((224, 224), Image.ANTIALIAS)
        aug_images = []
        for i in range(0, 7):
            temp_imgs = []
            temp_2 = temp.rotate(i * 45)
            temp_imgs.append(temp_2)
            temp_2.load()
            r, g, b = temp_2.split()
            temp_imgs.append(Image.merge('RGB', (r, b, g)))
            temp_imgs.append(Image.merge('RGB', (b, b, r)))
            temp_imgs.append(temp_2.convert(mode='L'))

            blur_range = [0, 2]
            for j in blur_range:
                for n in temp_imgs:
                    aug_images.append(n.filter(ImageFilter.BoxBlur(j)))

        try:
            for l in aug_images:
                l.save(os.path.join(path, str(count) + '.jpg'))
                count += 2

        except IOError:
            print('FAILURE TO SAVE IMAGE')
            continue
    print('\n')