from PIL import Image, ImageFilter
import os
from pathlib import Path
import function_sandbox as helper


cats_aug_dir = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Data\Dogs_Cats\Augment\cats')
dogs_aug_dir = Path(r'C:\Users\NG-ML\Johns_sandbox\charlie_data\Data\Dogs_Cats\Augment\dogs')

classes = [cats_aug_dir, dogs_aug_dir]
for individual_class in classes:
    base_name = str(os.path.basename(os.path.normpath(individual_class)))
    image_paths = helper.get_sub_dirs(individual_class)
    path = individual_class
    num_images = len(image_paths)
    count = 10
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
        for i in range(0, 3):
            temp_imgs =[]
            temp_2 = temp.rotate(i*90)
            temp_2.load()
            r, g, b = temp_2.split()
            temp_imgs.append(Image.merge('RGB', (b, r, b)))
            temp_imgs.append(Image.merge('RGB', (r, r, b)))
            temp_imgs.append(Image.merge('RGB', (r, b, g)))
            temp_imgs.append(Image.merge('RGB', (b, b, r)))
            temp_imgs.append(temp_2.convert(mode='L'))
            aug_images.append(temp_2.filter(ImageFilter.EMBOSS))
            aug_images.append(temp_2.filter(ImageFilter.FIND_EDGES))


            blur_range = [0, 2, 4]
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
