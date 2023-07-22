import numpy as np

# List your files here
file_list = ["1.npz", "2.npz", "3.npz", "4.npz", "5.npz", 
             "6.npz", "7.npz", "8.npz", "9.npz", "10.npz"]

# This list will hold all the images
all_images = []

for file in file_list:
    # Load each file
    path = "/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs_balanced_imgs/"
    data = np.load(path + file)

    # Append the 'images' array from each file to all_images
    all_images.append(data['images'])

# Stack all images arrays into one along the first axis
all_images = np.concatenate(all_images, axis=0)

# Save all images into one .npz file
np.savez(path + "combined_file.npz", images=all_images)