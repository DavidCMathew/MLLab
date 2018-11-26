import math


def dataset_split(data, arc, val):
    # declare a list variable to store the newly split data-set
    newData = []
    # iterate through every record in the data-set and split the data-set
    for rec in data:
        if rec[arc] == val:
            reducedSet = list(rec[:arc])
            reducedSet.extend(rec[arc+1:])
            newData.append(reducedSet)
    # return the new list that has the data-set that is split on the selected attribute
    return newData


def calc_entropy(data):
    # Calculate the length of the data-set
    entries = len(data)
    labels = {}
    # Read the class labels from the data-set file into the dict object "labels"
    for rec in data:
        label = rec[-1]
        if label not in labels.keys():
            labels[label] = 0
        labels[label] += 1
    # entropy variable is initialized to zero
    entropy = 0.0
    # For every class label (x) calculate the probability p(x)
    for key in labels:
        prob = float(labels[key])/entries
        # Entropy formula calculation
        entropy -= prob * math.log(prob, 2)
    # Return the entropy of the data-set
    return entropy


def attribute_selection(data):
    # get the number of features available in the given data-set
    features = len(data[0]) - 1

    # Fun call to calculate the base entropy (entropy of the entire data-set)
    baseEntropy = calc_entropy(data)

    # initialize the info-gain variable to zero
    max_InfoGain = 0.0
    bestAttr = -1

    # iterate through the features identified
    for i in range(features):
        # store the values of the features in a variable
        AttrList = [rec[i] for rec in data]

        # get the unique values from the feature values
        uniqueVals = set(AttrList)

        # initializing the entropy and the attribute entropy to zero
        newEntropy = 0.0
        attrEntropy = 0.0

        # iterate through the list of unique values and perform split
        for value in uniqueVals:

            # function call to split the data-set
            newData = dataset_split(data, i, value)

            # probability calculation
            prob = len(newData)/float(len(data))

            # entropy calculation for the attributes
            newEntropy = prob * calc_entropy(newData)
            attrEntropy += newEntropy

            # calculation of Information Gain
        infoGain = baseEntropy - attrEntropy
        print(infoGain)
        # identify the attribute with max info-gain
        if infoGain > max_InfoGain:
            max_InfoGain = infoGain
            bestAttr = i

    # return the attribute identified
    return bestAttr


def decision_tree(data, labels):
    # list variable to store the class-labels (terminal nodes of decision tree)
    classList = [rec[-1] for rec in data]

    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # functional call to identify the attribute for split
    maxGainNode = attribute_selection(data)

    # variable to store the class label value
    treeLabel = labels[maxGainNode]

    # dict object to represent the nodes in the decision tree
    theTree = {treeLabel:{}}
    del(labels[maxGainNode])

    # get the unique values of the attribute identified
    nodeValues = [rec[maxGainNode] for rec in data]
    uniqueVals = set(nodeValues)
    for value in uniqueVals:
        subLabels = labels[:]

        # update the non-terminal node values of the decision tree
        theTree[treeLabel][value] = decision_tree(dataset_split(data, maxGainNode, value),subLabels)

    # return the decision tree (dict object)
    return theTree


with open('data/tennis.csv', 'r') as csvfile:
    fdata = [line.strip() for line in csvfile]
    metadata = fdata[0].split(',')
    train_data = [x.split(',') for x in fdata[1:]]

tree = decision_tree(train_data, metadata)

print(tree)


