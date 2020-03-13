# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purpacman_poses provided that (1) you do not distribute or publish
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
import sys

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

        The evaluation function takes in the current and propacman_posed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman pacman_position after moving (newpacman_pos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newpacman_pos = successorGameState.getPacmanpacman_position()
        newFood = successorGameState.getFood()
        newghost_states = successorGameState.getghost_states()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newghost_states]

        "*** YOUR CODE HERE ***"

        newFoodList = newFood.asList()
        newFoodDistances = [manhattanDistance(f, newpacman_pos) for f in newFoodList]
        maxFoodDist = max(newFoodDistances) if len(newFoodDistances) > 0 else 0
        minFoodDist = min(newFoodDistances) if len(newFoodDistances) > 0 else 0

        newGhostpacman_positions = [g.getpacman_position() for g in newghost_states]
        newGhostDistances = [manhattanDistance(g, newpacman_pos) for g in newGhostpacman_positions]
        minGhostDist = min(newGhostDistances) if len(newGhostDistances) > 0 else 0

        newfood_count = newFood.count()

        score = -maxFoodDist + minGhostDist - newfood_count

        if minGhostDist <= 1:  # guarantee winning
            return -1000

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
        agent_index = self.index
        init_score = -999999
        first_move = None

        for action in gameState.getLegalActions(agent_index):
            successor = gameState.generateSuccessor(agent_index,action)
            curr_score = self.get_minimax_score(successor,1,0)
            if init_score < curr_score:
                first_move = action
                init_score = curr_score
        return first_move


    def get_minimax_score(self,state,agent_index,current_depth):
        if current_depth == self.depth and agent_index == 0:
            return self.evaluationFunction(state)

        if agent_index == 0:
            return self.maximize(state,agent_index,current_depth)
        else:
            return self.minimize(state,agent_index,current_depth)

    def maximize(self,state,agent_index,depth):
        score = -999999
        available_actions = state.getLegalActions(agent_index)
        next_agent = (agent_index + 1) % state.getNumAgents()
        if len(available_actions) == 0:
            return self.evaluationFunction(state)

        for action in available_actions:
            successor_state = state.generateSuccessor(agent_index, action)
            curr_score = self.get_minimax_score(successor_state, next_agent, depth)
            if curr_score > score:
                score = curr_score
        return score


    def minimize(self,state,agent_index,depth):
        score = 999999
        available_actions = state.getLegalActions(agent_index)
        next_agent = (agent_index + 1) % state.getNumAgents()
        next_depth = depth

        if len(available_actions) == 0:
            return self.evaluationFunction(state)

        if agent_index == state.getNumAgents() - 1:
            next_depth = depth + 1

        for action in available_actions:
            successor_state = state.generateSuccessor(agent_index, action)
            curr_score = self.get_minimax_score(successor_state, next_agent, next_depth)
            if curr_score < score:
                score = curr_score
        return score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agent_index = self.index
        init_score = -999999
        first_move = None

        alpha = -9999999
        beta = 999999
        for action in gameState.getLegalActions(agent_index):
            successor = gameState.generateSuccessor(agent_index, action)
            curr_score = self.get_minimax_score(successor, 1, 0,alpha,beta)
            alpha = max(alpha, curr_score)
            if init_score < curr_score:
                first_move = action
                init_score = curr_score
        return first_move

    def get_minimax_score(self, state, agent_index, current_depth, alpha, beta):
        if current_depth == self.depth and agent_index == 0:
            return self.evaluationFunction(state)

        if agent_index == 0:
            return self.maximize(state, agent_index, current_depth, alpha, beta)
        else:
            return self.minimize(state, agent_index, current_depth, alpha, beta)

    def maximize(self, state, agent_index, depth, alpha, beta):
        score = -999999
        available_actions = state.getLegalActions(agent_index)
        next_agent = (agent_index + 1) % state.getNumAgents()
        if len(available_actions) == 0:
            return self.evaluationFunction(state)

        for action in available_actions:
            successor_state = state.generateSuccessor(agent_index, action)
            curr_score = self.get_minimax_score(successor_state, next_agent, depth, alpha, beta)
            if curr_score > score:
                score = curr_score
            if score > beta:
                return score
            alpha = max(alpha, score)

        return score

    def minimize(self, state, agent_index, depth, alpha, beta):
        score = 999999
        available_actions = state.getLegalActions(agent_index)
        next_agent = (agent_index + 1) % state.getNumAgents()
        next_depth = depth

        if len(available_actions) == 0:
            return self.evaluationFunction(state)

        if agent_index == state.getNumAgents() - 1:
            next_depth = depth + 1

        for action in available_actions:
            successor_state = state.generateSuccessor(agent_index, action)
            curr_score = self.get_minimax_score(successor_state, next_agent, next_depth, alpha, beta)
            if curr_score < score:
                score = curr_score
            if score < alpha:
                return curr_score
            beta = min(beta, score)

        return score

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
        agent_index = self.index
        init_score = -999999
        first_move = None

        for action in gameState.getLegalActions(agent_index):
            successor = gameState.generateSuccessor(agent_index, action)
            curr_score = self.get_minimax_score(successor, 1, 0)
            if init_score < curr_score:
                first_move = action
                init_score = curr_score
        return first_move

    def get_minimax_score(self, state, agent_index, current_depth):
        if current_depth == self.depth and agent_index == 0:
            return self.evaluationFunction(state)

        if agent_index == 0:
            return self.maximize(state, agent_index, current_depth)
        else:
            return self.get_arbitrary_move_score(state, agent_index, current_depth)

    def maximize(self, state, agent_index, depth):
        score = -999999
        available_actions = state.getLegalActions(agent_index)
        next_agent = (agent_index + 1) % state.getNumAgents()
        if len(available_actions) == 0:
            return self.evaluationFunction(state)

        for action in available_actions:
            successor_state = state.generateSuccessor(agent_index, action)
            curr_score = self.get_minimax_score(successor_state, next_agent, depth)
            if curr_score > score:
                score = curr_score
        return score

    def get_arbitrary_move_score(self, state, agent_index, depth):

        available_actions = state.getLegalActions(agent_index)
        next_agent = (agent_index + 1) % state.getNumAgents()
        next_depth = depth

        if len(available_actions) == 0:
            return self.evaluationFunction(state)
        initial_score = 0
        if agent_index == state.getNumAgents() - 1:
            next_depth = depth + 1

        for action in available_actions:
            successor_state = state.generateSuccessor(agent_index, action)
            curr_score = self.get_minimax_score(successor_state, next_agent, next_depth)
            initial_score = initial_score + curr_score
        return initial_score / len(available_actions)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    pacman_pos = currentGameState.getPacmanpacman_position()
    food_list = currentGameState.getFood().asList()
    ghost_states = currentGameState.getghost_states()
    capsules = currentGameState.getCapsules()
    "*** YOUR CODE HERE ***"
    food_count = len(food_list)
    capsule_count = len(capsules)

    min_distance_near_food = 5000
    for food in food_list:
        distance_to_food = manhattanDistance(pacman_pos, food)

        if distance_to_food < min_distance_near_food:
            min_distance_near_food = distance_to_food
    near_ghost = 0

    min_dist_to_ghost = 5000
    for ghost in ghost_states:
        ghostpacman_pos = ghost.configuration.getpacman_position()
        distance_to_ghost = manhattanDistance(pacman_pos, ghostpacman_pos)

        if distance_to_ghost < min_dist_to_ghost:
            min_dist_to_ghost = distance_to_ghost
            near_ghost = ghost

    death_penalty = 0
    if min_dist_to_ghost == 0:
        if near_ghost.scaredTimer==0:
            death_penalty = 200
        min_dist_to_ghost = 1

    multiplier_ghost_distance = -2

    if near_ghost.scaredTimer > 0:
        multiplier_ghost_distance = near_ghost.scaredTimer / float(min_dist_to_ghost)

    if (min_dist_to_ghost-near_ghost.scaredTimer) > 0:
        multiplier_ghost_distance = 0


    return 1/float(min_distance_near_food)+multiplier_ghost_distance/float(min_dist_to_ghost)-food_count-10*capsule_count-death_penalty





# Abbreviation
better = betterEvaluationFunction
