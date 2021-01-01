import numpy as np
import sys
import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
from collections import Counter

class Packet:

    def __init__ (self, node, data, indices):
        self.node = node
        self.error = sum(np.where(compare(data, indices, node.label, 'Cover_Type', 3), 1, 0))
        self.truth = sum(np.where(compare(data, indices, node.label, 'Cover_Type', 2), 1, 0))
        self.child1 = None
        self.child2 = None
        if node.child != None:
            indices = [np.where(compare(data, indices, node.value, node.att, i), indices, -1) for i in range(3)]
            indices = [indices[i][indices[i] >= 0] for i in range(3)]
            if node.child == node.child1: indices[0] = np.append(indices[0],indices[2])
            else: indices[1] = np.append(indices[1],indices[2])
            self.child1 = Packet(node.child1, data, indices[0])
            self.child2 = Packet(node.child2, data, indices[1])

    def find (self):
        if self.node.child == None:
            return -1, self, self.error
        max1, packet1, agg_error1 = self.child1.find()
        max2, packet2, agg_error2 = self.child2.find()
        agg_error = agg_error1 + agg_error2
        maximum, priority, packet = max([(max1, 1, packet1), (max2, 2, packet2), (agg_error - self.error, 3, self)])
        return maximum, packet, agg_error

class Node:

    def __init__ (self, data, indices, att_type, entrp, label):
        self.entrp = entrp
        self.label = label
        self.att = None
        self.value = None
        self.child1 = None
        self.child2 = None
        self.child = None
        if len(att_type) != 0:
            att_type_new = att_type.copy()
            mdn = dict([(att, np.median(data[indices][att])) for att in att_type.keys() if att_type[att] == 'Continuous'])
            d_entrp = [self.divide(data, indices, mdn, att, att_type_new) for att in att_type.keys()]
            d_entrp, att, value, indices, entrp, label, child_index = max(d_entrp, key=lambda k: k[0])
            if d_entrp > 0:
                if att_type_new[att] == 'Discrete': att_type_new.pop(att)
                child = [Node(data, indices[i], att_type_new, entrp[i], label[i]) for i in range(2)]
                self.att = att
                self.value = value
                self.child1 = child[0]
                self.child2 = child[1]
                self.child = child[child_index]

    def divide (self, data, indices, mdn, att, att_type):
        if att_type[att] == 'Discrete':
            indices = [np.where(compare(data, indices, i, att, 2), indices, -1) for i in range(2)]
            indices = [indices[i][indices[i] >= 0] for i in range(2)]
            if len(indices[0]) == 0 or len(indices[1]) == 0:
                att_type.pop(att)
                return 0, None, None, None, None, None, None
            value = 0
            child_index = 0
        else:
            indices = [np.where(compare(data, indices, mdn[att], att, i), indices, -1) for i in range(3)]
            indices = [indices[i][indices[i] >= 0] for i in range(3)]
            if len(indices[2]) != 0 and len(indices[0]) == 0 and len(indices[1]) == 0:
                att_type.pop(att)
                return 0, None, None, None, None, None, None
            value = mdn[att]
            child_index = 0
            if len(indices[1]) == 0:
                indices[1] = indices[2]
                child_index = 1
            elif len(indices[0]) == 0: indices[0] = indices[2]
            elif len(indices[2]) != 0: indices[0] = np.append(indices[0],indices[2])
        count = [Counter(np.array([data[index] for index in indices[i]])['Cover_Type']) for i in range(2)]
        total = [sum(count[i].values()) for i in range(2)]
        entrp = [sum([entropy(freq, total[i]) for freq in count[i].values()]) for i in range(2)]
        label = [max(count[i].items(), key=lambda k: k[1])[0] for i in range(2)]
        res_entrp = (total[0] * entrp[0] + total[1] * entrp[1]) / (total[0] + total[1])
        d_entrp = self.entrp - res_entrp
        return d_entrp, att, value, indices, entrp, label, child_index

    def predict (self, sample):
        if self.child != None:
            if sample[self.att] < self.value: return self.child1.predict(sample)
            elif sample[self.att] > self.value: return self.child2.predict(sample)
            else: return self.child.predict(sample)
        else: return self.label

class DecisionTree:

    def __init__ (self, data, indices, att_type, entrp, label):
        self.head = Node(data, indices, att_type, entrp, label)

    def predict (self, data):
        return [self.head.predict(sample) for sample in data]

    def plot (self, data, indices, label):
        packet = Packet(self.head, data, indices)
        queue = [packet]
        y = [packet.truth]
        while len(queue) > 0:
            packet = queue.pop(0)
            if packet.node.child != None:
                queue.append(packet.child1)
                queue.append(packet.child2)
                y.append(y[-1] + packet.child1.truth + packet.child2.truth - packet.truth)
        plt.plot(np.arange(len(y))+1, 100 * np.array(y)/len(data), label = label +" "+ str(round(100 * y[-1] / len(data),2)) +"%")

    def prune (self, data, indices):
        head = Packet(self.head, data, indices)
        maximum, packet, agg_error = head.find()
        while True:
            maximum, packet, agg_error = head.find()
            if maximum < 0: break
            packet.child1 = None
            packet.child2 = None
            packet.node.att = None
            packet.node.value = None
            packet.node.child1 = None
            packet.node.child2 = None
            packet.node.child = None

def compare (data, indices, value, att, cmd):
    if cmd == 0: return [True if data[index][att] < value else False for index in indices]
    elif cmd == 1: return [True if data[index][att] > value else False for index in indices]
    elif cmd == 2: return [True if data[index][att] == value else False for index in indices]
    else: return [True if data[index][att] != value else False for index in indices]

def entropy (freq, total):
    p = freq / total
    if p == 0: return 0
    else: return -1 * p * np.log2(p)

file_train = open(sys.argv[2],"r")
line1_train = file_train.readline()
line2_train = file_train.readline()
att_type_train = dict([tuple(value.strip() for value in label.split(":")) for label in line1_train.split(",")])
att_type_train.pop('Cover_Type', None)
data_dt_train = np.dtype([(label.split(":")[0].strip(), np.int) for label in line1_train.split(",")])
data_train = np.array([tuple(cell for cell in row.split(",")) for row in file_train], dtype = data_dt_train)
indices_train = np.arange(len(data_train))
count_train = Counter(data_train['Cover_Type'])
total_train = sum(count_train.values())
entrp_train = sum([entropy(freq, total_train) for freq in count_train.values()])
label_train = max(count_train.items(), key=lambda k: k[1])[0]

file_val = open(sys.argv[3],"r")
line1_val = file_val.readline()
line2_val = file_val.readline()
att_type_val = dict([tuple(value.strip() for value in label.split(":")) for label in line1_val.split(",")])
att_type_val.pop('Cover_Type', None)
data_dt_val = np.dtype([(label.split(":")[0].strip(), np.int) for label in line1_val.split(",")])
data_val = np.array([tuple(cell for cell in row.split(",")) for row in file_val], dtype = data_dt_val)
indices_val = np.arange(len(data_val))

file_test = open(sys.argv[4],"r")
line1_test = file_test.readline()
line2_test = file_test.readline()
att_type_test = dict([tuple(value.strip() for value in label.split(":")) for label in line1_test.split(",")])
att_type_test.pop('Cover_Type', None)
data_dt_test = np.dtype([(label.split(":")[0].strip(), np.int) for label in line1_test.split(",")])
data_test = np.array([tuple(cell for cell in row.split(",")) for row in file_test], dtype = data_dt_test)
indices_test = np.arange(len(data_test))

dt = DecisionTree(data_train, indices_train, att_type_train, entrp_train, label_train)
if sys.argv[1] == "2":
    dt.prune(data_val, indices_val)
    plt.title("Decision Tree with Pruning")
else: plt.title("Decision Tree without Pruning")
plt.xlabel("Number of Nodes")
plt.ylabel("Accuracy")
dt.plot(data_train, indices_train, "Accuracy on Train Data =")
dt.plot(data_val, indices_val, "Accuracy on Validation Data =")
dt.plot(data_test, indices_test, "Accuracy on Test Data =")
plt.legend()
plt.savefig("decision_tree"+sys.argv[1]+".png")
plt.clf()
np.savetxt(sys.argv[5], dt.predict(data_test), fmt="%d", delimiter="\n")

# y_train = data_train['Cover_Type']
# y_val = data_val['Cover_Type']
# y_test = data_test['Cover_Type']
# data_train = np.array([[sample[att] for att in att_type_train.keys()] for sample in data_train])
# data_val = np.array([[sample[att] for att in att_type_train.keys()] for sample in data_val])
# data_test = np.array([[sample[att] for att in att_type_train.keys()] for sample in data_test])
#
# def forest (i, j, k, data, y):
#     random_forest = RandomForestClassifier(n_estimators = i, max_features = j, min_samples_split = k, bootstrap = True, oob_score = True)
#     random_forest.fit(data, y)
#     return random_forest.oob_score_, random_forest, [i, j, k]
#
# random_forest = [[[forest(i, j/10, k, data_train, y_train) for k in range(2, 12, 2)] for j in range(1, 11, 2)] for i in range(50, 550, 100)]
# parameter = max([max([max(random_forest[i][j], key=lambda k: k[0]) for j in range(5)], key=lambda k: k[0]) for i in range(5)], key=lambda k: k[0])[2]
# i = int((parameter[0]-50)/100)
# j = int((10*parameter[1]-1)/2)
# k = int((parameter[2]-2)/2)
# accuracy_oob = random_forest[i][j][k][1].oob_score_
# accuracy_train = random_forest[i][j][k][1].score(data_train, y_train)
# accuracy_val = random_forest[i][j][k][1].score(data_val, y_val)
# accuracy_test = random_forest[i][j][k][1].score(data_test, y_test)
#
# print(f"n_estimators = {parameter[0]}")
# print(f"max_features = {parameter[1]}")
# print(f"min_samples_split = {parameter[2]}")
# print(f"Out-Of-Bag Accuracy = {round(100 * accuracy_oob, 2)}%")
# print(f"Accuracy on Train Data = {round(100 * accuracy_train, 2)}%")
# print(f"Accuracy on Validation Data = {round(100 * accuracy_val, 2)}%")
# print(f"Accuracy on Test Data = {round(100 * accuracy_test, 2)}%")
#
# plt.title("max_features = "+str(parameter[1])+" & min_samples_split = "+str(parameter[2]))
# plt.xlabel("n_estimators")
# plt.ylabel("Accuracy")
# x = [50, 150, 250, 350, 450]
# plt.plot(x, [100 * random_forest[i][j][k][1].score(data_val, y_val) for i in range(5)], label = "Accuracy on Validation Data")
# plt.plot(x, [100 * random_forest[i][j][k][1].score(data_test, y_test) for i in range(5)], label = "Accuracy on Test Data")
# plt.legend()
# plt.savefig("n_estimators.png")
# plt.clf()
#
# plt.title("n_estimators = "+str(parameter[0])+" & min_samples_split = "+str(parameter[2]))
# plt.xlabel("max_features")
# plt.ylabel("Accuracy")
# x = [0.1, 0.3, 0.5, 0.7, 0.9]
# plt.plot(x, [100 * random_forest[i][j][k][1].score(data_val, y_val) for j in range(5)], label = "Accuracy on Validation Data")
# plt.plot(x, [100 * random_forest[i][j][k][1].score(data_test, y_test) for j in range(5)], label = "Accuracy on Test Data")
# plt.legend()
# plt.savefig("max_features.png")
# plt.clf()
#
# plt.title("n_estimators = "+str(parameter[0])+" & max_features = "+str(parameter[1]))
# plt.xlabel("min_samples_split")
# plt.ylabel("Accuracy")
# x = [2, 4, 6, 8, 10]
# plt.plot(x, [100 * random_forest[i][j][k][1].score(data_val, y_val) for k in range(5)], label = "Accuracy on Validation Data")
# plt.plot(x, [100 * random_forest[i][j][k][1].score(data_test, y_test) for k in range(5)], label = "Accuracy on Test Data")
# plt.legend()
# plt.savefig("min_samples_split.png")
# plt.clf()
