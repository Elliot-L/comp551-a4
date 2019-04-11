import pickle
def open1():
    pickleFile = open("filename.pickle", 'rb')
    file = pickle.load(pickleFile)
    list_of_arr = []
    list_of_targets = []
    for i in file:
        list_of_arr.append(i[0])
        list_of_targets.append(i[1])
    return list_of_arr,list_of_targets

print(open1())
