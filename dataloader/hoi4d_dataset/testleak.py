import os

def load_list_from_txt(path):
    datalist = []
    with open(path, "r") as f:
        for line in f:
                content = line.strip()
                if len(content) == 0:
                    continue
                datalist.append(content)
                
    return datalist

test_sum = {}
all_sum = {}

all_test = load_list_from_txt("dataloader/hoi4d_dataset/testset.txt")
for test in all_test:
    
    file_path = os.path.join('HOI4D', test, "objpose", "0.json")
    if not os.path.isfile(file_path):
        print("Missing file:", test)
    else:
        print('ok')
#     ele = test.split('/')
#     category = ele[2]
#     instance = ele[3][1:]
#     test_sum[category] = test_sum.get(category, set()) | set([instance])



# all = load_list_from_txt('dataloader/hoi4d_dataset/valid_arti_TrashCan.txt')
# for data in all:
#     ele = data.split('/')
#     category = ele[2]
#     instance = ele[3][1:]
#     all_sum[category] = all_sum.get(category, set()) | set([instance])



# print("all_test:", len(all_test))
# print("all:", len(all))