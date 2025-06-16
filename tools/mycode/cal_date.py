import json
import os
import config
from json.decoder import JSONDecodeError
from collections import defaultdict

if __name__ == "__main__":
    args = config.parse_args()
    metadata_path = args.metadata_path
    file_ls = os.listdir(metadata_path)
    date_files = defaultdict(list)
    for src_file in file_ls:
        try:
            with open(os.path.join(metadata_path,src_file),'r',encoding='utf-8') as file:
                data = json.load(file)
                date = data.get('date',None)
                if date:
                    date_key = f"{date['year']}"
                    date_files[date_key].append(src_file)
        except JSONDecodeError as e:
            print("JSONDecodeError occurred: {0}. Skipping file: {1}".format(e, src_file))
        except Exception as e:
            print("An error occurred: {0}. Skipping file: {1}".format(e, src_file))

    for date_key, files in date_files.items():
        print(f"Date: {date_key}, Files: {len(files)}")

