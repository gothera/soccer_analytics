import os

def count_jpg_files(directory="/Users/cosmincojocaru/Desktop/liga_1_dataset_1"):
    jpg_count = 0
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(directory):
        # Count files that end with .jpg or .jpeg (case insensitive)
        jpg_count += sum(1 for file in files if file.lower().endswith(('.jpg', '.jpeg')))
    
    return jpg_count

if __name__ == "__main__":
    total_jpgs = count_jpg_files()
    print(f"Total number of JPG files found: {total_jpgs}")