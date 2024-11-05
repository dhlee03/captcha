import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

# Paths for input and output directories
input_dir = "train"
output_dir = "processed"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def remove_lines(image):
    # Convert the image to grayscale if it's not already
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    img_array = np.array(gray_image)

    height, width = img_array.shape
    result = img_array.copy()  # Create a copy to modify

    for x in range(1, width - 1):
        for y in range(1, height - 1):
            count = 0
            pixel_value = img_array[y, x]

            # Check surrounding pixels
            if pixel_value == img_array[y - 1, x + 1]:
                count += 1
            if pixel_value == img_array[y, x + 1]:
                count += 1
            if pixel_value == img_array[y + 1, x + 1]:
                count += 1
            if pixel_value == img_array[y - 1, x]:
                count += 1
            if pixel_value == img_array[y + 1, x]:
                count += 1
            if pixel_value == img_array[y - 1, x - 1]:
                count += 1
            if pixel_value == img_array[y, x - 1]:
                count += 1
            if pixel_value == img_array[y + 1, x - 1]:
                count += 1

            # If the count is low, consider it noise and set it to a new value (e.g., white)
            if pixel_value == 0 and count <= 3 and count > 0:
                result[y, x] = 255  # Set to white to remove interference

    return result

def tokenize_contours(image):
    # Find contours of each character
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours
    filtered_contours = []
    areas = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        areas.append(w * h)

    # Filter contours based on area
    median_area = np.max(areas)
    for contour, area in zip(contours, areas):
        x, y, w, h = cv2.boundingRect(contour)
        if area > median_area / 10:
            filtered_contours.append((contour, x, y, w, h))
    
    # Sort contours from left to right based on x-coordinate
    filtered_contours = sorted(filtered_contours, key=lambda c: c[1])

    return filtered_contours

def tokenize_watershed(image, binary):
    # noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN,kernel, iterations = 2)
    
    # sure background area
    sure_bg = cv2.dilate(opening,  kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(image, markers)

    # return the box based on the markers
    boxes = []
    for i in range(1, markers.max() + 1):
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers == i] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        boxes.append((0, x, y, w, h))
    return boxes

def tokenize_projection(image, step=3):
    # Sum of white pixels along each column (for vertical segmentation)
    column_sums = np.sum(image, axis=0)

    # Find peaks in the column sums to detect character boundaries
    threshold = 0.2 * np.max(column_sums)
    start = None
    segments = []

    # Loop through the column sums with a step size
    for i in range(0, len(column_sums), step):
        sum_val = column_sums[i]
        
        if sum_val > threshold and start is None:
            start = i  # Start of a new character segment
        elif sum_val <= threshold and start is not None:
            segments.append((0, start, 0, i - start, image.shape[1]))  # End of a character segment
            start = None

    return segments

def divide_large_segment(segments, captcha_text):
    while len(segments) < len(captcha_text):
        # cannot divide none
        if len(segments) == 0:
            break
        # splits biggest contours
        idx = np.argmax(segments[:, 2])
        biggest_seg = segments[idx]
        # cannot split
        if biggest_seg[0] == 0:
            break
         # not always exactly half, can improve this part

        half_width = biggest_seg[2] // 2
        first_seg = np.array([biggest_seg[0], biggest_seg[1], half_width, biggest_seg[3]])
        second_seg = np.array([biggest_seg[0] + half_width, biggest_seg[1], biggest_seg[2] - half_width, biggest_seg[3]])
        segments = np.concatenate((segments[:idx], [first_seg, second_seg], segments[idx + 1:]))
    return segments

def process_image(file_path, file_name, charcount, tokenizor, output_dir):
    colored_img = cv2.imread(file_path)
    # Load the captcha image in grayscale
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # plt.imshow(img, cmap="gray")
    # plt.show()

    # Remove noisy lines
    img = remove_lines(img)

    # plt.imshow(img, cmap="gray")
    # plt.show()
    
    # Threshold the image to binary (black and white)
    _, processed_img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
    # _, processed_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # plt.imshow(processed_img, cmap="gray")
    # plt.show()

    # Invert the image to make the background white and the letters black
    processed_img = cv2.bitwise_not(processed_img)

    # Apply morphological closing to connect small gaps within characters
    kernel = np.ones((3, 3), np.uint8)  # Adjust kernel size based on your captcha structure
    processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)

    # Apply dilation to connect nearby parts of letters
    kernel = np.ones((2, 2), np.uint8)  # Adjust kernel size based on your captcha structure
    dilated_img = cv2.dilate(processed_img, kernel, iterations=1)

    # plt.imshow(dilated_img, cmap="gray")
    # plt.show()

    if tokenizor == 'contours':
        filtered_contours = tokenize_contours(dilated_img) 
    elif tokenizor == 'projection':
        filtered_contours = tokenize_projection(dilated_img)

    # Extract characters and save with the naming convention
    captcha_text = file_name.split("-")[0]  # Assuming filename format is 'text-0.png'

    charcount['total'] = charcount.get('total', 0) + len(captcha_text)

    dilation_steps = range(5)

    if len(captcha_text) != len(filtered_contours):
        # Apply dilation to connect nearby parts of letters
        for i in dilation_steps:
            dilated_img = cv2.dilate(processed_img, kernel, iterations=i)
            if tokenizor == 'contours':
                filtered_contours = tokenize_contours(dilated_img)
                if len(captcha_text) == len(filtered_contours):
                    break
            elif tokenizor == 'projection':
                for step in range(1, 5):
                    filtered_contours = tokenize_projection(dilated_img, step)
                    if len(captcha_text) == len(filtered_contours):
                        break

    filtered_contours = np.array([[x, y, w, h] for _, x, y, w, h in filtered_contours])
    if len(captcha_text) != len(filtered_contours):
        filtered_contours = divide_large_segment(filtered_contours, captcha_text)

    if len(captcha_text) != len(filtered_contours):
        # plt.imshow(dilated_img, cmap="gray")
        # plt.show()
        # print(f"Skipping {file_name} due to length mismatch: {len(captcha_text)} != {len(filtered_contours)}")
        return
    
    for i, (x, y, w, h) in enumerate(filtered_contours):
        # Crop and resize each character
        char_img = processed_img[y:y+h, x:x+w]
        char_img = cv2.resize(char_img, (224, 224))  # Resize to a fixed size
        
        # Save each character with the naming convention, mark the number of the character
        charcount[captcha_text[i]] = charcount.get(captcha_text[i], 0) + 1
        char_filename = f"{captcha_text[i]}_{charcount[captcha_text[i]]}.png"
        cv2.imwrite(os.path.join(output_dir, char_filename), char_img)

def prepare_image_folder(source, destination):
    # create the main folder if it doesn't exist
    os.makedirs(destination, exist_ok=True)

    # create 36 subfolders with names 0-9 and a-z
    folders = [str(i) for i in range(0, 10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
    for folder in folders:
        os.makedirs(os.path.join(destination, folder), exist_ok=True)

    # iterate through all png files in the source folder and move them to the appropriate subfolder
    for file_name in os.listdir(source):
        if file_name.endswith('.png'):
            # get the first character of the file name and convert it to lowercase
            first_char = file_name[0].lower()
            
            # check if the character matches a folder name
            if first_char in folders:
                # move the file to the corresponding subfolder
                src_path = os.path.join(source, file_name)
                dest_path = os.path.join(destination, first_char, file_name)
                shutil.move(src_path, dest_path)

    print("images have been successfully distributed into their respective folders!")
    

if __name__ == "__main__":
    # Process each file in the input directory
    tokenizer = 'contours'
    # tokenizer = 'projection'
    output_dir = f"{output_dir}_{tokenizer}"
    charcount = {}
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".png"):
            process_image(os.path.join(input_dir, filename), filename, charcount, tokenizer, output_dir)

    print(len(os.listdir(output_dir)) / charcount['total'])

    prepare_image_folder(output_dir, f'processed_train_{tokenizer}')