# Initialize the lightly pre-trained model with various scenarios to base real-time off of
machineLearningModel = initializeModel(trainingData)

# Calculate reward, if correct action is taken reward positively, else negatively
def calculateReward(actionTaken):
    if safteyTest(actionTaken):
        return highReward
    else:
        return lowReward

def updateModel(currentState, actionTaken, reward):
    # Making an estimate of the quality of the current state and also the action pair
    currentQ = estimateQ(currentState, actionTaken)
    
    # Calculate the max Q Value for next action from the new state
    maxFutureQ = maximumQ(newState, allActions)
    
    # Finds new Q Value using the reward and the max future Q Value
    newQ = currentQ + learningRate * (reward + discountFactor * maxFutureQ - currentQ)
    
    # New model Q Value for current state and action
    updateQ(currentState, actionTaken, newQ)
    
    # Update the model based on real-time learnings
    machineLearningModel.updatePolicy(currentState, actionTaken, newQ)

# While the car is driving run the real time model
while carDriving():
    sensorData = loadSensorData()
    currentState = processData(sensorData)

    nextAction = decideAction(machineLearningModel, currentState)
    takeAction(nextAction)

    outcome = observeOutcome(nextAction)
    reward = calculateReward(outcome)
    
    # Update the model with new data and reward
    updateModel(currentState, nextAction, reward)
