import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def convert_xml_to_csv(xml_folder, csv_output_folder):
    xml_list = []
    
    for xml_file in glob.glob(os.path.join(xml_folder, '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_file = os.path.splitext(root.find('filename').text)[0] + '.png'

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            xml_list.append({
                'filename': image_file,
                'width': int(root.find('size')[0].text),
                'height': int(root.find('size')[1].text),
                'class': class_name,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })

    df = pd.DataFrame(xml_list)
    csv_output_file = os.path.join(csv_output_folder, f'{os.path.basename(xml_folder)}.csv')
    df.to_csv(csv_output_file, index=False)

def main():
    xml_folder_train = os.path.join(os.getcwd(), 'images', 'train')
    xml_folder_test = os.path.join(os.getcwd(), 'images', 'test')
    csv_output_folder = os.path.join(os.getcwd(), 'images')
    
    convert_xml_to_csv(xml_folder_train, csv_output_folder)
    convert_xml_to_csv(xml_folder_test, csv_output_folder)
    
    print('Successfully converted XML annotations to CSV for train and test.')

if __name__ == '__main__':
    main()
