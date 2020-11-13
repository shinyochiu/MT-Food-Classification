import os
import shutil

def reorganize_data(data_path='../data', image_folder='train'):
    main_path = os.path.join(data_path, image_folder)
    image_list = os.listdir(main_path)
    print('Now processing path:', main_path)
    num_total = len(image_list)
    c = 0
    for im in image_list:
        label = im.split('_')[0]
        file_path = os.path.join(main_path, im)
        if os.path.isfile(file_path):
            target_folder = os.path.join(main_path, label)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            shutil.move(os.path.join(main_path, im), target_folder)
            c += 1
            if c % 500 == 0:
                print('process', c, '/', num_total)
    print('process', c, '/', num_total)
    print('finished')
    

if __name__ == "__main__":
    reorganize_data(data_path='../data', image_folder='train')
    reorganize_data(data_path='../data', image_folder='val')
