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
import random, util, math

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        newPos = successorGameState.getPacmanPosition() #(2,1)
        newFood = successorGameState.getFood() #FFFFTTTFF,FFFTFFF
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #[0]

        "*** YOUR CODE HERE ***"

        """
        Set the score to be large if the action leads to a winning state
        """
        if successorGameState.isWin():
        	return float("inf")
        
        score = successorGameState.getScore()#Initialize the score

        """
        Consider ghosts:
        	The score gets higher when the pacman-ghost distance becomes larger
        """
        ghostBase = -300
        curGhostPos = currentGameState.getGhostPosition(1)
        newGhostPos = successorGameState.getGhostPosition(1)

        if newPos in [newGhostPos,curGhostPos]:
        	score += ghostBase
        else:
          disToGhost = util.manhattanDistance(newPos, newGhostPos)
          score += ghostBase * (1.0 / disToGhost)

        if util.manhattanDistance(newPos, newGhostPos) < 2:
        	score -= float("inf")

        """
        Consider food:
        	1) If this action eats food, add 100 to the score;
			    2) If this action doesn't eat food, the score will be larger if the distance 
			   from pacman to the closest food is smaller(i.e. minus the distance to the score)
        """
        foodBase = 100
        foodList = newFood.asList()
        if newPos in foodList:
        	score += foodBase
        else:
          for food in foodList:
            disToFood = util.manhattanDistance(newPos, food)
            score -= foodBase * (1 - math.exp(-1.0 * 0.2 * disToFood))
     


        """
        Consider capsules:
        	The score gets higher when the action eats a capsule
        """
        capsulesBase = 150
        if newPos in currentGameState.getCapsules():
        	score += capsulesBase

        """
        Stop penalty
        """
        stopBase = -100
        if action == Directions.STOP:
          score += stopBase

        return score

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
    
    """
    This function is Depth-First implementation of Minimax
    """
    def DFMiniMax(self, gameState, depth, agent):
      
      best_move = Directions.STOP
      #Stop when all the moves are used up or it is a terminal node.
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState), best_move
      else:
        value = 0
        actions = gameState.getLegalActions(agent)
        
        if agent == 0:#Pacman(MAX)
          value = -float("inf")
          for action in actions:
            nextState = gameState.generateSuccessor(agent, action)
            #Next move is ghost with agentIndex 1(MIN)
            nxt_value, nxt_move = self.DFMiniMax(nextState, depth - 1, 1)
            if nxt_value > value:
              value, best_move = nxt_value, action
        else:#Ghosts(MIN)
          value = float("inf")
          for action in actions:
            nextState = gameState.generateSuccessor(agent, action)
            if agent == gameState.getNumAgents() - 1:#The last ghost
              nxt_value, nxt_move = self.DFMiniMax(nextState, depth - 1, 0)
            else:
              #The depth will not change since it has not finished one MIN step.
              #Agent will increase 1, representing the next ghost.
              nxt_value, nxt_move = self.DFMiniMax(nextState, depth, agent + 1)
            if nxt_value < value:
              value, best_move = nxt_value, action
        
        return value, best_move

    
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
        """
        "*** YOUR CODE HERE ***"

        #Multiply 2 to self.depth because MAX and MIN each gets self.depth moves to play.
        #The first player is Pacman(agentIndex == 0)
        best_action = self.DFMiniMax(gameState, self.depth * 2, 0)[1]

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    
    def AlphaBeta(self, gameState, depth, agent, alpha, beta):
      
      best_move = Directions.STOP
      #Stop when all the moves are used up or it is a terminal node.
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState), best_move
      else:
        value = 0
        actions = gameState.getLegalActions(agent)
        
        if agent == 0:#Pacman(MAX)
          value = -float("inf")
          for action in actions:
            nextState = gameState.generateSuccessor(agent, action)
            #Next move is ghost with agentIndex 1(MIN)
            nxt_value, nxt_move = self.AlphaBeta(nextState, depth - 1, 1, alpha, beta)
            if nxt_value > value:
              value, best_move = nxt_value, action
            if value >= beta:#Pruning condition satisfied
              return value, best_move
            alpha = max(alpha, value)#Update alpha
        else:#Ghosts(MIN)
          value = float("inf")
          for action in actions:
            nextState = gameState.generateSuccessor(agent, action)
            if agent == gameState.getNumAgents() - 1:#The last ghost
              nxt_value, nxt_move = self.AlphaBeta(nextState, depth - 1, 0, alpha, beta)
            else:
              #The depth will not change since it has not finished one MIN step.
              #Agent will increase 1, representing the next ghost.
              nxt_value, nxt_move = self.AlphaBeta(nextState, depth, agent + 1, alpha, beta)
            if nxt_value < value:
              value, best_move = nxt_value, action
            if value <= alpha:#Pruning condition satisfied
              return value, best_move
            beta = min(beta, value)#Update beta
        
        return value, best_move



    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #Multiply 2 to self.depth because MAX and MIN each gets self.depth moves to play.
        #The first player is Pacman(agentIndex == 0).
        #The initial value of alpha is -infinity and beta is infinity.
        best_action = self.AlphaBeta(gameState, self.depth * 2, 0, -float("inf"), float("inf"))[1]
        
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def Expectimax(self, gameState, depth, agent):
      
      best_move = Directions.STOP
      #Stop when all the moves are used up or it is a terminal node.
      if depth == 0 or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState), best_move
      else:
        value = 0
        actions = gameState.getLegalActions(agent)
        
        if agent == 0:#Pacman(MAX)
          value = -float("inf")
          for action in actions:
            nextState = gameState.generateSuccessor(agent, action)
            #Next move is CHANCE with agentIndex 1
            nxt_value, nxt_move = self.Expectimax(nextState, depth - 1, 1)
            if nxt_value > value:
              value, best_move = nxt_value, action
        else:#CHANCE player
          value = 0
          for action in actions:
            nextState = gameState.generateSuccessor(agent, action)
            if agent == gameState.getNumAgents() - 1:#The last CHANCE player
              nxt_value, nxt_move = self.Expectimax(nextState, depth - 1, 0)
            else:
              #Agent will increase 1, representing the next CHANCE player.
              nxt_value, nxt_move = self.Expectimax(nextState, depth, agent + 1)
            value += 1.0 / len(actions) * nxt_value #No best move for CHANCE layer
        
        return value, best_move

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        best_action = self.Expectimax(gameState, self.depth * 2, 0)[1]

        return best_action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      The final score is a linear combination of 6 components:
        1)current game score
        2)the number of left food dots
        3)the number of left capsules
        4)the minimum distance to left capsules(if any) 
        5)the minimum distance to left food dots(if any)
        6)the minimum distance to active ghosts(if any)
      Component 1, 6 has positive coefficients. 
      Components 2, 3, 4, 5, 6 have negative coefficients.

    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
      return float('inf')
    if currentGameState.isLose():
      return -float('inf')
    
    gameScore = currentGameState.getScore()#Initialize the score ######
    currPos = currentGameState.getPacmanPosition()#Current Pacman position
    ghostStates = currentGameState.getGhostStates()

    foodList = currentGameState.getFood().asList()
    foodNum = len(foodList)####
    minFoodDist = float('inf')####
    for food in foodList:
      currFoodDist = util.manhattanDistance(currPos, food)
      minFoodDist = min(minFoodDist, currFoodDist)

    capsulesList = currentGameState.getCapsules()
    capsulesNum = len(capsulesList)####
    minCapDist = float('inf')####
    for capsules in capsulesList:
      currCapDist = util.manhattanDistance(currPos, capsules)
      minCapDist = min(minCapDist, currCapDist)

    minActiveGhostDist = float('inf')####
    for ghost in ghostStates:
      if not ghost.scaredTimer:
        currGhostDist = util.manhattanDistance(currPos, ghost.getPosition())
        minActiveGhostDist = min(minActiveGhostDist, currGhostDist)
    
    score = gameScore  +\
           -4.0 * foodNum + \
           -20 * capsulesNum + \
           -1.0 * minCapDist +\
           -1.5 * minFoodDist + \
           1.0 * minActiveGhostDist

    return score

# Abbreviation
better = betterEvaluationFunction

