from attackDopel import *

toClass1 = targets_Task1.drop(['Unnamed: 0', 'index'], axis=1)
toClass2 = targets_Task2.drop(['Unnamed: 0', 'index'], axis=1)
toClass3 = targets_Task3.drop(['Unnamed: 0', 'index'], axis=1)
toClass4 = targets_Task4.drop(['Unnamed: 0', 'index'], axis=1)

def export_TargetsClassification(pathToResultsFolder, taskNumber, classifiers_collection, predictions, title=""):
    separator = "\n----------------------------------------------------------------------------------------------------\n"
    resultDateTime = dt.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    resultName = "results_Task" + str(taskNumber) + "_[" + resultDateTime + "].txt"
    resultPath = pathToResultsFolder + "Task " + str(taskNumber) + "/" + resultName
    resultTitle = title + "Results of classifiers for Task" + str(taskNumber) + "(" + resultDateTime + ")"
    resultsFile = open(resultPath, 'w')

    resultsFile.writelines(resultTitle)
    resultsFile.writelines(separator)

    modelIndex = 0
    for model in classifiers_collection:
        resultsFile.writelines(model.__class__.__name__ + " : \n-----\n")
        for i in range(len(predictions[modelIndex])):
            resultsFile.writelines(str(predictions[modelIndex][i]) + "\n")
        modelIndex += 1

    return 0

def classifyTarget(taskNumber, classifier_collection):
    predictions = []
    toClass = targetToClassifyFromInt(taskNumber)

    for model in classifier_collection:
        predictions.append(model.predict(toClass.values))
    return predictions

def targetToClassifyFromInt(taskNumber):
    match taskNumber:
        case 1:
            return toClass1
        case 2:
            return toClass2
        case 3:
            return toClass3
        case 4:
            return toClass4
        case _:
            return "Incorrect task number"