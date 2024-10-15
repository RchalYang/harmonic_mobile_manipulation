
def find_dining_table(house):
    count = 0
    for item in house["objects"]:
        if "Dining_Table" in item["assetId"]:
            count += 1
        if "dining_table" in item["assetId"]:
            count += 1
    # print(count)
    return count


def filter_dataset_cleaning_table(dataset):
    house_with_dining_table_list = []
    for idx, data_sample in enumerate(dataset):
        if len(data_sample["rooms"]) > 2:
            continue
        if find_dining_table(data_sample) == 1:
            house_with_dining_table_list.append(idx)

    room_with_cleaning_table_dataset = []
    import copy
    for idx in house_with_dining_table_list:
        data_sample = copy.deepcopy(dataset[idx])
        for i in range(len(data_sample["objects"])):
            if "Dining_Table" in data_sample["objects"][i]["assetId"]:
                data_sample["objects"][i]["assetId"] = "CleaningTable"
                data_sample["objects"][i]["id"] = "TableToClean"
                data_sample["objects"][i]["children"] = []

            if "dining_table" in data_sample["objects"][i]["assetId"]:
                data_sample["objects"][i]["assetId"] = "CleaningTable"
                data_sample["objects"][i]["id"] = "TableToClean"
                data_sample["objects"][i]["children"] = []
            #     continue

        for i in reversed(range(len(data_sample["objects"]))):
            if "Chair" in data_sample["objects"][i]["assetId"] or "chair" in data_sample["objects"][i]["assetId"]:
                data_sample["objects"].remove(data_sample["objects"][i])
        room_with_cleaning_table_dataset.append(data_sample)

    return room_with_cleaning_table_dataset
