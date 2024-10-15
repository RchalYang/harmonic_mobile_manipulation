import numpy as np
def find_fridge(house):
    count = 0
    for item in house["objects"]:
        if "Fridge" in item["assetId"]:
            count += 1
        if "fridge" in item["assetId"]:
            count += 1
    # print(count)
    return count

def filter_dataset_fridge(dataset):
    house_with_fridge_list = []
    for idx, data_sample in enumerate(dataset):
        if len(data_sample["rooms"]) > 2:
            continue
        if find_fridge(data_sample) == 1:
            house_with_fridge_list.append(idx)

    room_with_cleaning_table_dataset = []
    import copy
    for idx in house_with_fridge_list:
        data_sample = copy.deepcopy(dataset[idx])
        for i in range(len(data_sample["objects"])):
            if "Fridge" in data_sample["objects"][i]["assetId"]:
                data_sample["objects"][i]["assetId"] = "Fridge_22"
                data_sample["objects"][i]["id"] = "FridgeToOpen"
                data_sample["objects"][i]["children"] = []

            if "fridge" in data_sample["objects"][i]["assetId"]:
                data_sample["objects"][i]["assetId"] = "Fridge_22"
                data_sample["objects"][i]["id"] = "FridgeToOpen"
                data_sample["objects"][i]["children"] = []
                continue

        # for i in reversed(range(len(data_sample["objects"]))):
        #     if "Chair" in data_sample["objects"][i]["assetId"] or "chair" in data_sample["objects"][i]["assetId"]:
        #         data_sample["objects"].remove(data_sample["objects"][i])
        room_with_cleaning_table_dataset.append(data_sample)

    return room_with_cleaning_table_dataset


def get_fridge_property(scene):
    # door = scene["doors"][0]
    fridge = None
    for item in scene["objects"]:
        if "Fridge" in item["assetId"]:
            assert fridge is None
            fridge = item

    ori = fridge["rotation"]["y"]
    pos = np.array([
        fridge["position"]["x"],
        fridge["position"]["z"],
    ])

    # direction = (wall_corner_sec - wall_corner)
    direction = np.array([
        np.cos(np.deg2rad(ori)), np.sin(np.deg2rad(ori))
    ])

    opened_direction = np.array([
        np.sin(np.deg2rad(ori)), np.cos(np.deg2rad(ori))
    ])

    fridge_corner = pos + opened_direction * 0.37 - direction * 0.36

    return {
        "fridge_asset_id": fridge["assetId"],
        "fridge_id": fridge["id"],
        "fridge_corner": fridge_corner,
        "fridge_width": 0.72,
        "closed_direction": direction,
        "opened_direction": opened_direction,
        "fridge_closed_center": fridge_corner + 0.5 * - 0.72 * direction
    }
