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
from pacman import GameState

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
        newGhostPoss = [ngs.getPosition() for ngs in newGhostStates]
        #print(newPos,newGhostPoss,newScaredTimes)
        features = [0]*5 #if eat a ghost, if eat a food, 1/distance to closet next food, 1/distance of closet ghost, die
        weights = [2,1.5,1,1,-100]
        if newPos in newGhostPoss: #if agent and one ghost are at the same position in next state
            #print(newPos,newGhostPoss)
            #print(successorGameState.isLose())
            #print(newScaredTimes)
            ghost_ind = next(i for i in range(len(newGhostPoss)) if newGhostPoss[i]==newPos)
            ghost_scared_time = newScaredTimes[ghost_ind]
            if ghost_scared_time > 0: # the ghost still scared, then we eat the ghost
                #print("EAT")
                features[0] = 1 #this never occurred, I guess, if a ghost is eaten, then it is not a ghost any longer
                                #and a new one is born immediately, so you cannot judge whether you eat the ghost or not
                                #if newPos in newGhostPoss, the only case is that pacman is eaten
                                #also can use isLose() to check
            else: # the ghost is not scared, then the pacman is eaten, dead
                features[-1] = 1
        current_foods = currentGameState.getFood().asList()
        if newPos in current_foods:
            features[1] = 1 # eat a food
        new_food_distance = sorted([util.manhattanDistance(f,newPos) for f in newFood.asList()])
        if new_food_distance:
            features[2] = 1.0/new_food_distance[0]
        newGhostDis = [util.manhattanDistance(g,newPos) for g in newGhostPoss]
        if newGhostDis:
            minDis = min(newGhostDis)
            if minDis>0:
                minDisGhostInds = [i for i in range(len(newGhostDis)) if newGhostDis[i]==minDis]
                minDisGhostScaredTimes = [newScaredTimes[i] for i in minDisGhostInds]
                if any(st<minDis for st in minDisGhostScaredTimes):
                    features[3] = -1.0/minDis # if any closet ghost has no enough scared time, then the agent is in danger
                else:
                    features[3] = 1.0/minDis #otherwise, there is no danger
        return sum(w*f for w,f in zip(weights,features))



        #"*** YOUR CODE HERE ***"
        #return successorGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
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
        def value(gameState,step):
            if step//number_agents == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), 'Stop'
            ind = step%number_agents
            if ind == 0:
                return maxValue(gameState,step)
            else:
                return minValue(gameState,step)

        def maxValue(gameState,step):
            agent_id =  step%number_agents
            legal_actions = gameState.getLegalActions(agent_id)
            max_value = float("-inf")
            max_action = None
            for action in legal_actions:
                next_state = gameState.generateSuccessor(agent_id,action)
                v,_ = value(next_state,step+1)
                if v > max_value:
                    max_value = v
                    max_action = action
            return max_value, max_action

        def minValue(gameState,step):
            agent_id = step%number_agents
            legal_actions = gameState.getLegalActions(agent_id)
            min_value = float("inf")
            min_action = None
            for action in legal_actions:
                next_state = gameState.generateSuccessor(agent_id,action)
                v,_ = value(next_state,step+1)
                if v < min_value:
                    min_value = v
                    min_action = action
            return min_value, min_action

        number_agents = gameState.getNumAgents()
        v,a =  value(gameState,0)
        #print(v)
        return a

        #"*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def value(gameState,step,alpha,beta):
            if step//number_agents == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), 'Stop'
            ind = step%number_agents
            if ind == 0:
                return maxValue(gameState,step,alpha,beta)
            else:
                return minValue(gameState,step,alpha,beta)

        def maxValue(gameState,step,alpha,beta):
            agent_id =  step%number_agents
            legal_actions = gameState.getLegalActions(agent_id)
            max_value = float("-inf")
            max_action = None
            for action in legal_actions:
                next_state = gameState.generateSuccessor(agent_id,action)
                v,_ = value(next_state,step+1,alpha,beta)
                if v > max_value:
                    max_value = v
                    max_action = action
                if max_value > beta:
                    return max_value,max_action
                alpha = max(alpha,max_value)
            return max_value, max_action

        def minValue(gameState,step,alpha,beta):
            agent_id = step%number_agents
            legal_actions = gameState.getLegalActions(agent_id)
            min_value = float("inf")
            min_action = None
            for action in legal_actions:
                next_state = gameState.generateSuccessor(agent_id,action)
                v,_ = value(next_state,step+1,alpha,beta)
                if v < min_value:
                    min_value = v
                    min_action = action
                if min_value < alpha:
                    return min_value,min_action
                beta = min(beta,min_value)
            return min_value, min_action

        number_agents = gameState.getNumAgents()
        v,a =  value(gameState,0,float("-inf"),float("inf"))
        #print(v)
        return a
        #"*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

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
        def value(gameState,step):
            if step//number_agents == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), 'Stop'
            ind = step%number_agents
            if ind == 0:
                return maxValue(gameState,step)
            else:
                return expValue(gameState,step)

        def maxValue(gameState,step):
            agent_id =  step%number_agents
            legal_actions = gameState.getLegalActions(agent_id)
            max_value = float("-inf")
            max_action = None
            for action in legal_actions:
                next_state = gameState.generateSuccessor(agent_id,action)
                v,_ = value(next_state,step+1)
                if v >= max_value:
                    max_value = v
                    max_action = action
            return max_value, max_action

        def expValue(gameState,step):
            agent_id = step%number_agents
            legal_actions = gameState.getLegalActions(agent_id)
            ave_value = 0
            #min_action = None
            for action in legal_actions:
                next_state = gameState.generateSuccessor(agent_id,action)
                v,_ = value(next_state,step+1)
                ave_value += v
            return ave_value/len(legal_actions), None

        number_agents = gameState.getNumAgents()
        v,a =  value(gameState,0)
        #print(v)
        return a

        #"*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState:GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    features = [0.]*8
    weights = [20000,10,1,1,45,0.5,3,1]
    #1/number_food,1/cloest_food_dis,
    #1/num_pallets, 1/cloest_pallet, if_empty_pallet(encourage to eat the last pallet), scared_time(encourage the pacman to eat pallet to increase scared_time)
    #encourage to eat the ghost, some scared time becomes 0, some are not
    #1/distance_to_ghost
    #weights = []
    if currentGameState.isWin():
        #print(currentGameState.getFood().asList())
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")
    pac_pos = currentGameState.getPacmanPosition()
    food_pos = currentGameState.getFood().asList()
    num_food = len(food_pos)
    ghost_states = currentGameState.getGhostStates()
    ghost_scared_times = [ghostState.scaredTimer for ghostState in ghost_states]
    ghost_pos = [g.getPosition() for g in ghost_states]
    pallets = currentGameState.getCapsules()
    features[0] = 1/num_food
    food_dis = sorted([util.manhattanDistance(fp,pac_pos) for fp in food_pos])
    features[1] = 1/food_dis[0]
    pallets_dis = sorted([util.manhattanDistance(p,pac_pos) for p in pallets])
    if pallets_dis:
        features[2] = 1/len(pallets_dis)
        features[3] = 1/pallets_dis[0]
    features[4] = int(not pallets) # if pallets become zero, we need to encourage this
    features[5] = sum(ghost_scared_times)
    features[6] = 1 if 0 in ghost_scared_times and sum(ghost_scared_times)>0 else 0
    # if some ghost's scared time becomes zero, some are not, then some ghost is eaten
    # for those states after the ghost is eaten, this will always be 1, but does not afftect the action selection, because we compare the relative relation
    ghost_dis = [util.manhattanDistance(g, pac_pos) for g in ghost_pos]
    minDis = min(ghost_dis)
    min_dis_ghost_inds = [i for i in range(len(ghost_dis)) if ghost_dis[i] == minDis]
    min_dis_ghost_scared_times = [ghost_scared_times[i] for i in min_dis_ghost_inds]
    if any(st < minDis for st in min_dis_ghost_scared_times):
        features[7] = -1.0 / minDis  # if any closet ghost has no enough scared time, then the agent is in danger
    else:
        features[7] = 1.0 / minDis  # otherwise, there is no danger
    return sum(w*f for w,f in zip(weights,features))
    #"*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
