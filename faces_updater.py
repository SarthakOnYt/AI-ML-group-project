import os
import json

def main():
    images=os.listdir('images')

    with open("faces.json", "r") as ri:
        data=json.load(ri)

    for image in images:
        if image.endswith('.jpg') and f"images/{image}" not in data:
            print(f"Adding {image} to faces.json")
            data[f"images/{image}"] = {"authorized": "Yes"}
    
    with open("faces.json", "w") as wi:
        json.dump(data, wi, indent=4)

if __name__ == "__main__":
    main()