import os
import json

def main():

    #add images in mages folder to faces.json file
    images=os.listdir('images')

    with open("faces.json", "r") as ri:
        data=json.load(ri)

    for image in images:
        if image.endswith('.jpg') and f"images/{image}" not in data:
            print(f"Adding {image} to faces.json")
            data[f"images/{image}"] = {"authorized": "Yes"}

    #remove the unnecessary json files which do not exist in the images folder

    keys_to_remove=[file for file in data if (file.startswith("images/") and file[7:] not in os.listdir("images"))]
    for file in keys_to_remove:
            del data[file]
    
    with open("faces.json", "w") as wi:
        json.dump(data, wi, indent=4)

if __name__ == "__main__":
    main()