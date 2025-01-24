import os
import pandas as pd

from data.teamExperiments.attackDopel import npzPrivateToDataframe, publicData_Tasks12, publicData_Tasks34

def createFinalTrainingSet(nonMembersPart, membersPart):
    addMembershipToSubset(nonMembersPart, 0)
    addMembershipToSubset(membersPart, 1)
    trainingSet = pd.concat([nonMembersPart, membersPart])
    return trainingSet

def makeMemberPart(privateDataset_Path, clean = False):
    members_clean = pd.DataFrame()
    members_unclean = pd.DataFrame()
    setToExclude = []
    for file in os.listdir(privateDataset_Path):
        # print(file)
        if file.endswith(".npz"):
            members_unclean = pd.concat([members_unclean, npzPrivateToDataframe(os.path.join(privateDataset_Path, file))], ignore_index=True)
            if file not in setToExclude:
                members_clean = pd.concat([members_clean, npzPrivateToDataframe(os.path.join(privateDataset_Path, file))], ignore_index=True)
    if clean:
        return members_clean
    else:
        return members_unclean
    

def makeNonMemberPart(memberPart, taskNumber):
    nonMembers = pd.DataFrame()
    if taskNumber == 1 or taskNumber == 2:
        allNonMembers = pd.concat([publicData_Tasks12, memberPart, memberPart]).drop_duplicates(keep=False)
    elif taskNumber == 3 or taskNumber == 4:
        allNonMembers = pd.concat([publicData_Tasks34, memberPart, memberPart]).drop_duplicates(keep=False)
    else:
        print("Task number not recognized")
        return 1
    while len(nonMembers) < len(memberPart):
        rowToAdd = allNonMembers.sample()
        nonMembers = pd.concat([nonMembers, rowToAdd])
    return nonMembers

def makeSubsetFromDataset(dataFrame, subsetStartIndex, subsetEndIndex_excluded):
    # Make small datasets from the original dataset
    return dataFrame.iloc[subsetStartIndex:subsetEndIndex_excluded].copy()

def addMembershipToSubset(subset, isMember):
    newSubset = subset.copy()
    if isMember == 0 or isMember == 1:
        subset["isMember"] = isMember
        return newSubset
    else:
        print("Second parameter must be 0 or 1. Your dataset hasn't changed")
        return newSubset