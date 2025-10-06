## ID: 20240614 NAME: Lee, Jongwon
######################################################################################
# Problem 2a
# minimax value of the root node: 5
# pruned edges (in order): h, m, t, x
######################################################################################

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
from collections.abc import Iterable

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    AGENT_NUM = gameState.getNumAgents()
    PACMAN_INDEX = self.index
    DEPTH = self.depth
    INITIAL_LEGAL_ACTIONS = gameState.getLegalActions(PACMAN_INDEX)

    assert DEPTH >= 0, "Depth should be non-negative integer."
    assert INITIAL_LEGAL_ACTIONS, "There should be at least one legal action."
    assert AGENT_NUM >= 1, "There should be at least one agent."
    assert 0 <= PACMAN_INDEX < AGENT_NUM, "Pacman index out of bounds."
    assert callable(self.evaluationFunction), "evaluationFunction should be callable."
    assert type(self.evaluationFunction(gameState)) in (int, float), "evaluationFunction should return a number."

    if len(INITIAL_LEGAL_ACTIONS) == 1:
      return INITIAL_LEGAL_ACTIONS.pop()

    def get_next_agent_index(agentIndex: int) -> int:
      return (agentIndex + 1) % AGENT_NUM

    stack: list[tuple[int, int, GameState, bool, int]] = []
    returnStack: list[int | float] = []

    nextAgentIndex = get_next_agent_index(PACMAN_INDEX)
    nextDepth = 1 if nextAgentIndex == PACMAN_INDEX else 0
    for action in INITIAL_LEGAL_ACTIONS:
      successor = gameState.generateSuccessor(PACMAN_INDEX, action)
      stack.append((nextAgentIndex, nextDepth, successor, False, 0))

    while stack:
      agentIndex, depth, gameState, isRetState, retNums = stack.pop()

      if isRetState:
        compareFunc = max if agentIndex == PACMAN_INDEX else min
        retVal = compareFunc((returnStack.pop() for _ in range(retNums)))
      elif gameState.isWin() or gameState.isLose() or depth >= DEPTH:
        retVal = self.evaluationFunction(gameState)
      else:
        legalActions = gameState.getLegalActions(agentIndex)

        if not legalActions:
          retVal = self.evaluationFunction(gameState)
        else:
          nextAgentIndex = get_next_agent_index(agentIndex)
          nextDepth = (depth + 1) if nextAgentIndex == PACMAN_INDEX else depth

          stack.append((agentIndex, depth, gameState, True, len(legalActions)))
          for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            stack.append((nextAgentIndex, nextDepth, successor, False, 0))

          continue

      returnStack.append(retVal)

    if len(returnStack) != len(INITIAL_LEGAL_ACTIONS):
      raise RuntimeError("Something went wrong while searching minimax tree.")

    score, action = max(zip(returnStack, reversed(INITIAL_LEGAL_ACTIONS)))
    return action
    

######################################################################################
# Problem 2b: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    AGENT_NUM = gameState.getNumAgents()
    PACMAN_INDEX = self.index
    DEPTH = self.depth
    INITIAL_LEGAL_ACTIONS = gameState.getLegalActions(PACMAN_INDEX)

    INF = float('inf')
    N_INF = float('-inf')

    assert DEPTH >= 0, "Depth should be non-negative integer."
    assert INITIAL_LEGAL_ACTIONS, "There should be at least one legal action."
    assert AGENT_NUM >= 1, "There should be at least one agent."
    assert 0 <= PACMAN_INDEX < AGENT_NUM, "Pacman index out of bounds."
    assert callable(self.evaluationFunction), "evaluationFunction should be callable."
    assert type(self.evaluationFunction(gameState)) in (int, float), "evaluationFunction should return a number."

    if len(INITIAL_LEGAL_ACTIONS) == 1:
      return INITIAL_LEGAL_ACTIONS.pop()

    def get_next_agent_index(agentIndex: int) -> int:
      return (agentIndex + 1) % AGENT_NUM
    
    stack: list[tuple[int, int, GameState, bool, int | float, int | float, list[str] | None]] = []
    returnStack: list[int | float] = []

    nextAgentIndex = get_next_agent_index(PACMAN_INDEX)
    nextDepth = 1 if nextAgentIndex == PACMAN_INDEX else 0
    for action in INITIAL_LEGAL_ACTIONS:
      successor = gameState.generateSuccessor(PACMAN_INDEX, action)
      stack.append((nextAgentIndex, nextDepth, successor, False, N_INF, INF, None))

    while stack:
      agentIndex, depth, gameState, isRetState, alpha, beta, legalActions = stack.pop()

      isMaxLayer = agentIndex == PACMAN_INDEX
      
      if isRetState:
        compFunc = max if isMaxLayer else min
        a, b = returnStack.pop(), returnStack.pop()
        retVal = compFunc(a, b)

        if legalActions:
          if (isMaxLayer and retVal < beta) or (not isMaxLayer and retVal > alpha):
            if isMaxLayer:
              alpha = max(alpha, retVal)
            else:
              beta = min(beta, retVal)

            nextAgentIndex = get_next_agent_index(agentIndex)
            nextDepth = (depth + 1) if nextAgentIndex == PACMAN_INDEX else depth

            action = legalActions.pop()
            successor = gameState.generateSuccessor(agentIndex, action)

            stack.append((agentIndex, depth, gameState, True, alpha, beta, legalActions))
            stack.append((nextAgentIndex, nextDepth, successor, False, alpha, beta, None))
      elif gameState.isWin() or gameState.isLose() or depth >= DEPTH:
        retVal = self.evaluationFunction(gameState)
      else:
        legalActions = gameState.getLegalActions(agentIndex)

        if not legalActions:
          retVal = self.evaluationFunction(gameState)
        else:
          nextAgentIndex = get_next_agent_index(agentIndex)
          nextDepth = (depth + 1) if nextAgentIndex == PACMAN_INDEX else depth
          
          action = legalActions.pop()
          successor = gameState.generateSuccessor(agentIndex, action)

          stack.append((agentIndex, depth, gameState, True, alpha, beta, legalActions))
          stack.append((nextAgentIndex, nextDepth, successor, False, alpha, beta, None))

          retVal = N_INF if isMaxLayer else INF

      returnStack.append(retVal)

    if len(returnStack) != len(INITIAL_LEGAL_ACTIONS):
      raise RuntimeError("Something went wrong while searching minimax tree.")
    
    score, action = max(zip(returnStack, reversed(INITIAL_LEGAL_ACTIONS)))
    return action

######################################################################################
# Problem 3a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState: GameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    AGENT_NUM = gameState.getNumAgents()
    PACMAN_INDEX = self.index
    DEPTH = self.depth
    INITIAL_LEGAL_ACTIONS = gameState.getLegalActions(PACMAN_INDEX)

    assert DEPTH >= 0, "Depth should be non-negative integer."
    assert INITIAL_LEGAL_ACTIONS, "There should be at least one legal action."
    assert AGENT_NUM >= 1, "There should be at least one agent."
    assert 0 <= PACMAN_INDEX < AGENT_NUM, "Pacman index out of bounds."
    assert callable(self.evaluationFunction), "evaluationFunction should be callable."
    assert type(self.evaluationFunction(gameState)) in (int, float), "evaluationFunction should return a number."

    if len(INITIAL_LEGAL_ACTIONS) == 1:
      return INITIAL_LEGAL_ACTIONS.pop()

    def get_next_agent_index(agentIndex: int) -> int:
      return (agentIndex + 1) % AGENT_NUM
    
    def mean(values: Iterable[int | float]) -> float:
      vals = list(values)
      return sum(vals) / len(vals)

    stack: list[tuple[int, int, GameState, bool, int]] = []
    returnStack: list[int | float] = []

    nextAgentIndex = get_next_agent_index(PACMAN_INDEX)
    nextDepth = 1 if nextAgentIndex == PACMAN_INDEX else 0
    for action in INITIAL_LEGAL_ACTIONS:
      successor = gameState.generateSuccessor(PACMAN_INDEX, action)
      stack.append((nextAgentIndex, nextDepth, successor, False, 0))

    while stack:
      agentIndex, depth, gameState, isRetState, retNums = stack.pop()

      if isRetState:
        compareFunc = max if agentIndex == PACMAN_INDEX else mean
        retVal = compareFunc((returnStack.pop() for _ in range(retNums)))
      elif gameState.isWin() or gameState.isLose() or depth >= DEPTH:
        retVal = self.evaluationFunction(gameState)
      else:
        legalActions = gameState.getLegalActions(agentIndex)

        if not legalActions:
          retVal = self.evaluationFunction(gameState)
        else:
          nextAgentIndex = get_next_agent_index(agentIndex)
          nextDepth = (depth + 1) if nextAgentIndex == PACMAN_INDEX else depth

          stack.append((agentIndex, depth, gameState, True, len(legalActions)))
          for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            stack.append((nextAgentIndex, nextDepth, successor, False, 0))

          continue

      returnStack.append(retVal)

    if len(returnStack) != len(INITIAL_LEGAL_ACTIONS):
      raise RuntimeError("Something went wrong while searching minimax tree.")

    score, action = max(zip(returnStack, reversed(INITIAL_LEGAL_ACTIONS)))
    return action

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float | int:
  """
  Your extreme, unstoppable evaluation function (problem 4).
  """
  INF = float('inf')
  epsilon = 0.01

  score = currentGameState.getScore()
  
  pacmanPos = currentGameState.getPacmanPosition()
  capsulesPos = currentGameState.getCapsules()
  foodsPos = currentGameState.getFood().asList()
  scaredGhosts = [ghost for ghost in currentGameState.getGhostStates() if ghost.scaredTimer >= 1]

  foodDists = [INF, *(manhattanDistance(pacmanPos, foodPos) for foodPos in foodsPos)]
  capsuleDists = [INF, *(manhattanDistance(pacmanPos, capPos) for capPos in capsulesPos)]
  scaredGhostDists = [INF, *(manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in scaredGhosts)]
  
  minFoodDist = min(foodDists) + epsilon
  minCapsuleDist = min(capsuleDists) + epsilon
  minScaredGhostDist = min(scaredGhostDists) + epsilon

  features = [1 / minFoodDist, 1 / minScaredGhostDist, score, len(foodsPos), len(capsulesPos)]
  weights = [9, 165, 1.3125, -9, -800]

  if not scaredGhosts:
    features.append(1 / minCapsuleDist)
    weights.append(550)

  finalScore = sum(map(lambda x: x[0] * x[1], zip(features, weights)))

  return finalScore

# Abbreviation
better = betterEvaluationFunction

