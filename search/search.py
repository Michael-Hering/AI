# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """

    class DFSStrategy():
      def __init__(self):
        self.frontier = util.Stack()
        self.childParent = {}

      def expandFrontier(self, currentState, explored, successorStates): # Expand the frontier given successor states
        validSuccessors = [state for state in successorStates if state[0] not in [e[0] for e in explored]] # Only keep states not in explored set        
        for state in validSuccessors:
          self.frontier.push(state)
          self.childParent[state] = currentState

      def getLeaf(self): # Return a leaf node for expansion
        leaf = self.frontier.pop() # Get the node on the top of the stack
        return leaf

    return genericSearch(problem, DFSStrategy())

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    class BFSStrategy():
      def __init__(self):
        self.frontier = util.Queue()
        self.childParent = {}

      def expandFrontier(self, currentState, explored, successorStates): # Expand the frontier given successor states
        validSuccessors = [state for state in successorStates if state[0] not in [e[0] for e in explored] and state[0] not in [s[0] for s in self.frontier.list]] # Only keep states not in frontier or explored set        
        for state in validSuccessors:
          self.frontier.push(state)
          self.childParent[state] = currentState

      def getLeaf(self): # Return a leaf node for expansion
        leaf = self.frontier.pop() # Get the next node from queue
        return leaf

    return genericSearch(problem, BFSStrategy())

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    
    class UCSStrategy():
      def __init__(self):
        self.frontier = util.PriorityQueue()
        self.childParent = {}

      def expandFrontier(self, currentState, explored, successorStates): # Expand the frontier given successor states      
        if (currentState == problem.getStartState()):
          for state in successorStates:
            self.childParent[state] = currentState
            self.frontier.push(state, state[2])
        else:
          for state in successorStates:
            check_f = [f for f in self.frontier.heap if state == f[2]]
            if ((state not in explored and len(check_f) == 0)):
              self.childParent[state] = currentState
              cost = 0
              curr = state
              while(curr != problem.getStartState()):
                cost += curr[2]
                curr = self.childParent[curr]

              self.frontier.update(state, cost)

      def getLeaf(self): # Return a leaf node for expansion
        leaf = self.frontier.pop() # Get the bbest node from the priority queue
        return leaf
    
    return genericSearch(problem, UCSStrategy())

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
      
    class AStarStrategy():
      def __init__(self):
        self.frontier = util.PriorityQueue()
        self.childParent = {}

      def expandFrontier(self, currentState, explored, successorStates): # Expand the frontier given successor states      
        if (currentState == problem.getStartState()):
          for state in successorStates:
            self.childParent[state] = currentState
            self.frontier.update(state, state[2] + heuristic(state[0], problem))
        else:
          validSuccessors = [state for state in successorStates if state[0] not in [e[0] for e in explored] and state not in [s for s in self.frontier.heap]] # Only keep states not in frontier or explored set        
          for state in validSuccessors:
            self.childParent[state] = currentState

            cost = 0
            curr = state
            while(curr != problem.getStartState()):
              cost += curr[2]
              curr = self.childParent[curr]

            cost += heuristic(state[0], problem)

            self.frontier.update(state, cost)
          
      def getLeaf(self): # Return a leaf node for expansion
        leaf = self.frontier.pop() # Get the best node from the priority queue
        return leaf

    return genericSearch(problem, AStarStrategy())

############################################################################
  # Check for whole state when checking if a state is ni the frontier #
############################################################################

def genericSearch(problem, strategy):
    solution = []
    explored = []
    frontier = strategy.frontier # Initialize the frontier according to strategy
    startSuccessors = problem.getSuccessors(problem.getStartState()) # Use the initial state of the problem
    explored.append(problem.getStartState()) # Initialize the explored set
    strategy.expandFrontier(problem.getStartState(), explored, startSuccessors)
    while(not frontier.isEmpty()):
      leaf = strategy.getLeaf() # Choose a leaf node for expansion according to strategy
      
      check_e = [ex for ex in explored if leaf[0] == ex[0]]
      if (len(check_e) > 0):
        continue
      
      if problem.isGoalState(leaf[0]): # Node is the goal state, build our solution
        curr = leaf
        while (curr != problem.getStartState()):
          solution.insert(0, curr[1])
          curr = strategy.childParent[curr]

        return solution
      
      explored.append(leaf) # Add the node to the explored set
      leafSuccessors = problem.getSuccessors(leaf[0]) # Get all of the successor states
      strategy.expandFrontier(leaf, explored, leafSuccessors) # Expand the frontier according to strategy

    return solution
      
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
