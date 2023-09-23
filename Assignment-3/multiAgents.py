# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def alphabeta(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            print(value)
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = alphabeta(gameState)

        return action

        # def minimax_search(state, agentIndex, depth):
        #     # if in min layer and last ghost
        #     if agentIndex == state.getNumAgents():
        #         # if reached max depth, evaluate state
        #         if depth == self.depth:
        #             return self.evaluationFunction(state)
        #         # otherwise start new max layer with bigger depth
        #         else:
        #             return minimax_search(state, 0, depth + 1)
        #     # if not min layer and last ghost
        #     else:
        #         moves = state.getLegalActions(agentIndex)
        #         # if nothing can be done, evaluate the state
        #         if len(moves) == 0:
        #             return self.evaluationFunction(state)
        #         # get all the minimax values for the next layer with each node being a possible state after a move
        #         next = (minimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

        #         # if max layer, return max of layer below
        #         if agentIndex == 0:
        #             return max(next)
        #         # if min layer, return min of layer below
        #         else:
        #             return min(next)
        # # select the action with the greatest minimax value
        # result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(gameState.generateSuccessor(0, x), 1, 1))

        # return result        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.minimax(gameState,0, self.depth)[1]
    
    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        actions = []
        for action in gameState.getLegalActions(agentIndex):
            v = self.minimax(gameState.generateSuccessor(agentIndex,action), agentIndex+1,depth, alpha, beta)[0]
            actions.append((v,action))
            if v > beta:
                return (v,action)
            alpha = max(alpha,v)
        return max(actions)
    
    def minValue(self,gameState, agentIndex, depth, alpha, beta):
        actions = []
        for action in gameState.getLegalActions(agentIndex):
            v = self.minimax(gameState.generateSuccessor(agentIndex,action),agentIndex+1,depth,alpha,beta)[0]
            actions.append((v,action))
            if v < alpha:
                return (v,action)
            beta = min(beta,v)
        return min(actions)
    
    def minimax(self,gameState,agentIndex,depth,alpha=-999999,beta=999999):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return (self.evaluationFunction(gameState),"Stop")
        
        agentsNum = gameState.getNumAgents()
        agentIndex %= agentsNum
        if agentIndex == agentsNum - 1:
            depth -=1
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.Expectimax(gameState, 0, self.depth)[1]

    def maxValue(self, gameState, agentIndex, depth):
        actions = []
        for action in gameState.getLegalActions(agentIndex):
            actions.append((self.Expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0], action))   
        return max(actions)

    def minValue(self, gameState, agentIndex, depth):
        actions = []
        total = 0
        for action in gameState.getLegalActions(agentIndex):
            v = self.Expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0]
            total += v
            actions.append((v, action))
        return (total / len(actions), )
    
    def Expectimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return ( self.evaluationFunction(gameState), "Stop")
        
        agentsNum = gameState.getNumAgents()
        agentIndex %=  agentsNum
        if agentIndex == agentsNum - 1:
            depth -= 1
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.minValue(gameState, agentIndex, depth)
        
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newGhostPos = currentGameState.getGhostPositions()
    # Cách khoảng cách từ pacman đến ma, chấm thức ăn, điểm chấm lớn và số lượng các đối tượng tương ứng, và khoảng cách gần nhất tới các đối tượng
    nearestGdist, gdist, nghost = 999999, 0, 0
    nearestFdist, fdist, nfood = 999999, 0, 0
    gscore, bonus, ncapsule, cdist = 0, 0, 0, 0
    # Nếu trạng thái của game là thắng thì trả về điểm số lớn nhất
    if currentGameState.isWin() ==  True:    
        return 999999
    # Nếu trạng thái của game là thua thì trả về điểm số nhỏ nhất
    elif currentGameState.isLose() == True:  
        return -999999
    # Tính khoảng cách manhattan từ pacman tới các con ma và tìm khoảng cách ngắn nhất
    for ghost in newGhostPos:
        nghost += 1
        mdist = manhattanDistance(ghost, newPos)
        gdist += mdist
        if (nearestGdist < mdist): nearestGdist = mdist
    # Tính khoảng cách manhattan từ pacman tới các điểm thức ăn và tìm khoảng cách ngắn nhất
    for food in newFood.asList():
        mdist = manhattanDistance(food, newPos)
        fdist += mdist
        if (nearestFdist < mdist): nearestFdist = mdist
        nfood += 1
    # Tạo biến điểm thưởng hay điểm ưu tiên có ảnh hưởng tới giá trị của một trạng thái
    bonus= 0
    # Tính điểm thưởng khi gần ma thì trừ điểm, gần chấm thức ăn thì tăng điểm
    if nearestGdist > 0 and nearestGdist < 3:
        bonus = -1200/nearestGdist
        if nearestGdist == 1: bonus -= 5000
    elif (nearestFdist > 0):
        bonus += 5000/nearestFdist
    # Tính điểm thưởng dựa trên số lượng thức ăn và khoảng cách chấm thức ăn
    if nfood < 4 and fdist > 0: bonus += 5000/fdist
    # Nếu có ma thì điểm sẽ tăng bằng số khoảng cách của các con ma chia cho bình phương số lượng ma
    if nghost != 0: gscore = gdist/(nghost)**2
    else: gscore = 800
    # Tính khoảng cách từ pacman tới các điểm chấm lớn và số lượng điểm chấm lớn 
    for capsule in newCapsules:
        cdist += manhattanDistance(capsule, newPos)
        ncapsule += 1
    
    # Chu kì khi ăn điểm chấm lớn và thời gian hoảng sợ của ma (ảnh hưởng tới quyết định đuổi theo ăn ma)
    duration = 2*sum(newScaredTimes)
    if (duration == 0):
        duration = 10 * cdist
    
    rs =  gscore - 7*fdist/nfood + 10*bonus + currentGameState.getScore() - duration - 5000*nfood
    return rs

    # 

    # closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
    # if newCapsules:
    #     closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
    # else:
    #     closestCapsule = 0

    # if closestCapsule:
    #     closest_capsule = -3 / closestCapsule
    # else:
    #     closest_capsule = 100

    # if closestGhost:
    #     ghost_distance = -2 / closestGhost
    # else:
    #     ghost_distance = -500

    # foodList = newFood.asList()
    # if foodList:
    #     closestFood = min([manhattanDistance(newPos, food) for food in foodList])
    # else:
    #     closestFood = 0

    # return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule

# Abbreviation
better = betterEvaluationFunction

'''
----MiniMax agent
Capsule classic [-463,-438,-430,-478,-477] --> Loss
Contest classic [-76,350,377,-164,-202] --> Loss
Medium classic [-883,1284(W),-3308(W),152,-589] --> 3/5 Loss
Minimax classic [-492,514(W),511(W),-496,511(W)] --> 2/5 Loss
Test Classic [516(W),550(W),544(W),538(W),528(W)] --> Win

----AlphaBeta agent
Capsule classic [80,-279,-367,-156,-130] --> Loss
Contest classic [-89,-360,-250,-256,252] --> Loss
Medium classic [-28,69,-29,129,-340] --> Loss
Minimax classic [511(W),-492,-495,512(W),-495] --> 3/5 Loss
Test Classic [490,516,516,536,520] --> Win

---- ExpectiMax agent
Capsule classic [-427,56,146,558,157] --> Loss
Contest classic [216,1094,563,-202,235] --> Loss
Medium classic [-361,1860(W),-1100,-956,-136] --> 4/5 Loss
Minimax classic [511(W),-501,511(W),-505,507(W)] --> 2/5 Loss
Test Classic [528(W),532(W),502(W),522(W),560(W)] ---> Win


------------------------with better evaluation function
----MiniMax agent
Capsule classic [1236(W),972(W),-625,-434,877(W)] --> 2/5 Loss
Contest classic [2210(W),1731,355,1607(W),2056(W)] --> 2/5 Loss  
Medium classic [1735(W),1390(W),1530(W),1806(W),1603(W)] --> Win 
Minimax classic [-492,-492,-492,-492,-492] --> Loss 
Test Classic [562(W),492(W),538(W),564(W),484(W)] --> Win 

----AlphaBeta agent
Capsule classic [-446,-481,793(W),815(W),-255] -->  3/5 Loss
Contest classic [2398(W),1556(W),2400(W),1113(W),1143(W)] --> Win
Medium classic [1783(W),1510(W),1906(W),1890(w),1793(W)] --> Win
Minimax classic [-492,510(W),509(W),512(W),510(W)] --> 1/5 Loss
Test Classic [536(W),564(W),564(W),554(W),366(W)] --> Win

---- ExpectiMax agent
Capsule classic [-179,-450,53,-452,1197(W)] --> 4/5 Loss 
Contest classic [1628(W),1848(W),1811(W),1621(W),2278(W)] --> Win
Medium classic [1690(W),868,1782(W),1488(W),1387(W)] --> 1/5 Loss
Minimax classic [-505,-494,515(W),515(W),508(W)] --> 2/5 Loss 
Test Classic [536(W),484(W),468(W),522(W),552(W)] ---> Win 
'''