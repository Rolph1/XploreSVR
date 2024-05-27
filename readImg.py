from PIL import Image
import os
import csv

requiredDimensions = [60, 70]  # width, height after cropping
crop_rectangle = (1025, 778, 1085, 848)

# crops the function since we need specific dimensions (ROI)
def crop_func(img):
    # crop rectangle (top right, bottom right, top left, bottom left
    #crop_rectangle = (1025, 778, 1600, 1579)
    return img.crop(crop_rectangle)


# gets rgb coordinates and puts them in a tuple
def get_rgb(image, requiredDimensions):
    rgb_values = []
    for x in range(requiredDimensions[0]):
        for y in range(requiredDimensions[1]):
            r, g, b = image.getpixel((x, y))
            rgb_values.append([r, g, b])
    return rgb_values


# writes the RGB values in a CSV file
def write_to_csv(rgb_values, filename):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rgb_values)

IMAGES_FOLDER = "C:\\Users\\PC\\Documents\\imagesToOpen"
csv_filename = 'data.csv'

# Write data to CSV
write_to_csv([['R', 'G', 'B']], csv_filename)

# Get list of image files
files = [os.path.join(IMAGES_FOLDER, f) for f in os.listdir(IMAGES_FOLDER) if os.path.isfile(os.path.join(IMAGES_FOLDER, f))]

counter = 0
file_name = "img_data"
for currFile in files:
    try:
        image = Image.open(currFile)
        image = crop_func(image)

        #if image.size != tuple(requiredDimensions):
        #    print(f"Skipping {currFile}: Wrong dimensions")
        #    continue

        rgb_image = image.convert('RGB')
        rgb_values = get_rgb(rgb_image, requiredDimensions)
        csv_name_2 = file_name + str(counter)
        write_to_csv(rgb_values, csv_name_2)
        counter += 1
    except Exception as e:
        print(f"Error processing {currFile}: {e}")

print(f"Processed {len(files)} images.")
