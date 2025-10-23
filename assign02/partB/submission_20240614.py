# ID: 20240614 NAME: Jongwon Lee
######################################################################################

from engine.const import Const
import util, math, random, collections, itertools


############################################################
# Problem 1: Warmup
def get_conditional_prob1(delta: float, epsilon: float, eta: float, c2: int, d2: int) -> float:
    """
    :param delta: [δ] is the parameter governing the distribution of the initial car's position
    :param epsilon: [ε] is the parameter governing the conditional distribution of the next car's position given the previos car's position
    :param eta: [η] is the parameter governing the conditional distribution of the sensor's measurement given the current car's position
    :param c2: the car's 2nd position
    :param d2: the sensor's 2nd measurement

    :returns: a number between 0~1 corresponding to P(C_2=c2 | D_2=d2)
    """
    DOMAIN = 0, 1
    
    P_c1 = delta, 1 - delta
    P_c_given_cp = lambda c, cp: epsilon if c != cp else 1 - epsilon
    P_d_given_c = lambda d, c: eta if d != c else 1 - eta

    P_c2 = tuple(map(lambda c2: sum(P_c1[c1] * P_c_given_cp(c2, c1) for c1 in DOMAIN), DOMAIN))
    P_d2_given_c2 = tuple(map(P_d_given_c, (d2, d2), DOMAIN))

    P_d2 = tuple(map(math.prod, zip(P_c2, P_d2_given_c2)))
    normalizer = sum(P_d2)

    return P_d2[c2] / normalizer


def get_conditional_prob2(delta: float, epsilon: float, eta: float, c2: int, d2: int, d3: int) -> float:
    """
    :param delta: [δ] is the parameter governing the distribution of the initial car's position
    :param epsilon: [ε] is the parameter governing the conditional distribution of the next car's position given the previos car's position
    :param eta: [η] is the parameter governing the conditional distribution of the sensor's measurement given the current car's position
    :param c2: the car's 2nd position
    :param d2: the sensor's 2nd measurement
    :param d3: the sensor's 3rd measurement

    :returns: a number between 0~1 corresponding to P(C_2=c2 | D_2=d2, D_3=d3)
    """
    DOMAIN = 0, 1

    P_c1 = delta, 1 - delta
    P_c_given_cp = lambda c, cp: epsilon if c != cp else 1 - epsilon
    P_d_given_c = lambda d, c: eta if d != c else 1 - eta

    P_c2 = tuple(map(lambda c2: sum(P_c1[c1] * P_c_given_cp(c2, c1) for c1 in DOMAIN), DOMAIN))
    P_d2_given_c2 = tuple(map(P_d_given_c, (d2, d2), DOMAIN))
    P_d3_given_c2 = tuple(map(lambda c2: sum(P_c_given_cp(c3, c2) * P_d_given_c(d3, c3) for c3 in DOMAIN), DOMAIN))
    
    P_d2d3_given_c2 = tuple(map(math.prod, zip(P_d2_given_c2, P_d3_given_c2)))
    P_d2d3 = tuple(map(math.prod, zip(P_c2, P_d2d3_given_c2)))
    normalizer = sum(P_d2d3)

    return P_d2d3[c2] / normalizer


def get_epsilon():
    """
    return a value of epsilon (ε)
    """
    return 0.5


# Class: ExactInference
# ---------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using exact updates (correct, but slow times).
class ExactInference(object):

    # Function: Init
    # --------------
    # Constructer that initializes an ExactInference object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows, numCols):
        self.skipElapse = (
            False  ### ONLY USED BY GRADER.PY in case problem 3 has not been completed
        )
        # util.Belief is a class (constructor) that represents the belief for a single
        # inference state of a single car (see util.py).
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()

    ############################################################
    # Problem 2:
    # Function: Observe (update the probablities based on an observation)
    # -----------------
    # Takes |self.belief| and updates it based on the distance observation
    # $d_t$ and your position $a_t$.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard
    #                 deviation Const.SONAR_STD
    #
    # Notes:
    # - Convert row and col indices into locations using util.rowToY and util.colToX.
    # - util.pdf: computes the probability density function for a Gaussian
    # - Don't forget to normalize self.belief!
    ############################################################

    def observe(self, agentX: float, agentY: float, observedDist: float):
        for r, c in itertools.product(range(self.belief.getNumRows()), range(self.belief.getNumCols())):
            ax, ay = util.colToX(c), util.rowToY(r)
            mu = math.dist((agentX, agentY), (ax, ay))
            prob = util.pdf(mu, Const.SONAR_STD, observedDist)
            prev_prob = self.belief.getProb(r, c)
            self.belief.setProb(r, c, prev_prob * prob)

        self.belief.normalize()


    ############################################################
    # Problem 3:
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Takes |self.belief| and updates it based on the passing of one time step.
    # Notes:
    # - Use the transition probabilities in self.transProb, which gives all
    #   ((oldTile, newTile), transProb) key-val pairs that you must consider.
    # - Other ((oldTile, newTile), transProb) pairs not in self.transProb have
    #   zero probabilities and do not need to be considered.
    # - util.Belief is a class (constructor) that represents the belief for a single
    #   inference state of a single car (see util.py).
    # - Be sure to update beliefs in self.belief ONLY based on the current self.belief distribution.
    #   Do NOT invoke any other updated belief values while modifying self.belief.
    # - Use addProb and getProb to manipulate beliefs to add/get probabilities from a belief (see util.py).
    # - Don't forget to normalize self.belief!
    ############################################################
    def elapseTime(self):
        if self.skipElapse:
            return  ### ONLY FOR THE GRADER TO USE IN Problem 2
        
        belief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols())
        for tile in itertools.product(range(self.belief.getNumRows()), range(self.belief.getNumCols())):
            prob = sum(
                self.belief.getProb(*oldTile) * transProb
                for (oldTile, newTile), transProb in self.transProb.items()
                if newTile == tile
            )
            belief.setProb(*tile, prob)
        
        self.belief = belief
        self.belief.normalize()

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.
    def getBelief(self):
        return self.belief


# Class: Particle Filter
# ----------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using a set of particles.
class ParticleFilter(object):

    NUM_PARTICLES = 200

    # Function: Init
    # --------------
    # Constructer that initializes an ParticleFilter object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows, numCols):
        self.belief = util.Belief(numRows, numCols)

        # Load the transition probabilities and store them in a dict of defaultdict
        # self.transProbDict[oldTile][newTile] = probability of transitioning from oldTile to newTile
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if oldTile == (2, 10):
                print(oldTile)
            if not oldTile in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(int)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]

        # Initialize the particles randomly
        self.particles = collections.defaultdict(int)
        potentialParticles = list(self.transProbDict.keys())
        for i in range(self.NUM_PARTICLES):
            particleIndex = int(random.random() * len(potentialParticles))
            self.particles[potentialParticles[particleIndex]] += 1

        self.updateBelief()

    # Function: Update Belief
    # ---------------------
    # Updates |self.belief| with the probability that the car is in each tile
    # based on |self.particles|, which is a defaultdict from particle to
    # probability (which should sum to 1).
    def updateBelief(self):
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief

    ############################################################
    # Problem 4 (part a):
    # Function: Observe:
    # -----------------
    # Takes |self.particles| and updates them based on the distance observation
    # $d_t$ and your position $a_t$.
    # This algorithm takes two steps:
    # 1. Reweight the particles based on the observation.
    #    Concept: We had an old distribution of particles, we want to update these
    #             these particle distributions with the given observed distance by
    #             the emission probability.
    #             Think of the particle distribution as the unnormalized posterior
    #             probability where many tiles would have 0 probability.
    #             Tiles with 0 probabilities (no particles), we do not need to update.
    #             This makes particle filtering runtime to be O(|particles|).
    #             In comparison, exact inference (problem 2 + 3), most tiles would
    #             would have non-zero probabilities (though can be very small).
    # 2. Resample the particles.
    #    Concept: Now we have the reweighted (unnormalized) distribution, we can now
    #             resample the particles and update where each particle should be at.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    #
    # Notes:
    # - Create |self.NUM_PARTICLES| new particles during resampling.
    # - To pass the grader, you must call util.weightedRandomChoice() once per new particle.
    ############################################################
    def observe(self, agentX: float, agentY: float, observedDist: float):
        def getWeightedCount(r: int, c: int, cnt: int) -> float:
            x, y = util.colToX(c), util.rowToY(r)
            mu = math.dist((agentX, agentY), (x, y))
            prob = util.pdf(mu, Const.SONAR_STD, observedDist)
            return cnt * prob

        weightedDict = {tile: getWeightedCount(*tile, cnt) for tile, cnt in self.particles.items()}
        self.particles = collections.defaultdict(int, collections.Counter(
            util.weightedRandomChoice(weightedDict) for _ in range(self.NUM_PARTICLES)
        ))

        self.updateBelief()

    ############################################################
    # Problem 4 (part b):
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Read |self.particles| (defaultdict) corresonding to time $t$ and writes
    # |self.particles| corresponding to time $t+1$.
    # This algorithm takes one step
    # 1. Proposal based on the particle distribution at current time $t$:
    #    Concept: We have particle distribution at current time $t$, we want to
    #             propose the particle distribution at time $t+1$. We would like
    #             to sample again to see where each particle would end up using
    #             the transition model.
    #
    # Notes:
    # - transition probabilities is now using |self.transProbDict|
    # - Use util.weightedRandomChoice() to sample a new particle.
    # - To pass the grader, you must loop over the particles using
    #       for tile in self.particles
    #   and call util.weightedRandomChoice() $once per particle$ on the tile.
    ############################################################
    def elapseTime(self):
        #for oldTile, cnt in self.particles.items():
            #print(oldTile)
            #print(self.transProbDict[oldTile])

        self.particles = collections.defaultdict(int, collections.Counter(
            newTile 
            for oldTile, cnt in self.particles.items() 
            if oldTile in self.transProbDict
            for newTile in map(util.weightedRandomChoice, [self.transProbDict[oldTile]] * cnt)
        ))
        self.updateBelief()

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.
    def getBelief(self):
        return self.belief
