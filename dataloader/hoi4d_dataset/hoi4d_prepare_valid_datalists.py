import os
import numpy as np

category2name_map = {
    "C1": "ToyCar",
    "C2": "Mug",
    "C3": "Laptop",
    "C4": "StorageFurniture",
    "C5": "Bottle",
    "C6": "Safe",
    "C7": "Bowl",
    "C8": "Bucket",
    "C9": "Scissors",
    "C11": "Pliers",
    "C12": "Kettle",
    "C13": "Knife",
    "C14": "TrashCan",
    "C17": "Lamp",
    "C18": "Stapler",
    "C20": "Chair"
}

no_ok = np.loadtxt('dataloader/hoi4d_dataset/not_ok_ins.txt', dtype=str)

manual_not_ok_ins = {
    "C3": ['N65'],
    "C4": ['N33', 'N34', 'N35', 'N36', 'N37', 'N38', 'N39', 'N40'],
    "C6": [],
    "C8": [],
    "C9": [],
    "C11": [],
    "C14": ['N01', 'N16', 'N17', 'N23', 'N26', 'N27', 'N28', 'N29', 'N33', 'N34', 'N35', 'N40', 'N44'],
    "C17": [],
    "C18": [],
}



target_cate = "C3"  # Safe
target_cate_name = category2name_map[target_cate]

not_ok_ins = []
for ins in no_ok:
    if target_cate_name in ins:
        not_ok_ins.append('N'+ins.split('/')[-1][1:])


def prepare_datalists(root_dir, testset=[]):
    datalist = []
    for ZY in os.listdir(root_dir):
        p1 = os.path.join(root_dir, ZY)
        if not os.path.isdir(p1):
            continue
        for H in os.listdir(p1):
            p2 = os.path.join(p1, H)
            if not os.path.isdir(p2):
                continue
            for C in os.listdir(p2):
                p3 = os.path.join(p2, C)
                if not os.path.isdir(p3) or C not in [target_cate]:
                    continue
                for N in os.listdir(p3):
                    
                    if N in not_ok_ins:
                        continue
                    if N in manual_not_ok_ins.get(C, []):
                        continue
                    
                    p4 = os.path.join(p3, N)
                    if not os.path.isdir(p4):
                        continue
                    for S in os.listdir(p4):
                        p5 = os.path.join(p4, S)
                        if not os.path.isdir(p5):
                            continue
                        for s in os.listdir(p5):
                            p6 = os.path.join(p5, s)
                            if not os.path.isdir(p6):
                                continue
                            for T in os.listdir(p6):
                                p7 = os.path.join(p6, T)
                                if not os.path.isfile(os.path.join(p7, "objpose", "0.json")):
                                    continue
                                
                                # if T not in ['T2']:
                                #     continue
                                
                            # _path = os.path.join(ZY, H, C, N, S, s, T)
                            # if _path in testset:
                            #     continue
                            #     ...
                                datalist.append(os.path.join(ZY, H, C, N, S, s, T))
    return datalist

def load_list_from_txt(path):
    datalist = []
    with open(path, "r") as f:
        for line in f:
                content = line.strip()
                if len(content) == 0:
                    continue
                datalist.append(content)
                
    return datalist

if __name__ == "__main__":
    root_dir = 'HOI4D'
    
    # allset = load_list_from_txt('all.txt')
    
    datalist = prepare_datalists(root_dir, )
    f = open(f"./valid_arti_{target_cate_name}.txt", "w")
    for d in datalist:
        f.write(d + "\n")
    f.close()
