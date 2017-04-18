from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        '''
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        '''

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            '''Create all concrete Load actions and return a list

            :return: list of Action objects
            '''
            loads = []
            # TODO create all load ground actions from the domain Load action
            # (i.e  cargos will move from airport into the plane)
            # (i.e plane at the airport already and will not leave the airport during load action)
            """
                Action : Defines an action schema using preconditions and effects
                Use this to describe actions in PDDL
                action is an Expr where variables are given as arguments(args)
                Precondition and effect are both lists with positive and negated literals
                Example:
                    precond_pos = [expr("Human(person)"), expr("Hungry(Person)")]
                    precond_neg = [expr("Eaten(food)")]
                    effect_add = [expr("Eaten(food)")]
                    effect_rem = [expr("Hungry(person)")]
                    eat = Action(expr("Eat(person, food)"), [precond_pos, precond_neg], [effect_add, effect_rem])
            """
            for cargo in self.cargos:
                for plane in self.planes:
                    for airport in self.airports:
                        precond_pos = [expr("At({}, {})".format(cargo, airport)),
                                      expr("At({}, {})".format(plane, airport)),]
                        precond_neg = []
                        effect_add = [expr("In({}, {})".format(cargo, plane)),] # effect : cargo in the plane
                        effect_rem = [expr("At({}, {})".format(cargo, airport)),] # effect : cargo not (leave) at the airport
                        load = Action(expr("Load({}, {}, {})".format(cargo, plane, airport)),
                                      [precond_pos, precond_neg],
                                      [effect_add, effect_rem])
                        loads.append(load)

            return loads

        def unload_actions():
            '''Create all concrete Unload actions and return a list

            :return: list of Action objects
            '''
            unloads = []
            # TODO create all Unload ground actions from the domain Unload action
            # (i.e  cargos will be moved from the plane to the airport)
            # (i.e plane at the airport already and will not leave the airport during unload action)
            for cargo in self.cargos:
                for plane in self.planes:
                    for airport in self.airports:
                        precond_pos = [expr("In({}, {})".format(cargo, plane)),
                                      expr("At({}, {})".format(plane, airport)),]
                        precond_neg = []
                        effect_add = [expr("At({}, {})".format(cargo, airport)),] # Effect : cargo  is  at the airport
                        effect_rem = [expr("In({}, {})".format(cargo, plane)),]   # Effect : cargo not in the plane
                        unload = Action(expr("Unload({}, {}, {})".format(cargo, plane, airport)),
                                      [precond_pos, precond_neg],
                                      [effect_add, effect_rem])
                        unloads.append(unload)

            return unloads


        def fly_actions():
            '''Create all concrete Fly actions and return a list

            :return: list of Action objects
            '''
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for plane in self.planes:
                            precond_pos = [expr("At({}, {})".format(plane, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(plane, to))] # Effect : plane in (flight) to destination airport
                            effect_rem = [expr("At({}, {})".format(plane, fr))] # Effect : plane not in (leave) original airport
                            fly = Action(expr("Fly({}, {}, {})".format(plane, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()




    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # TODO implement
        possible_actions = []

        '''
        for action in self.actions_list:
            # Checks if the precondition is satisfied in the current state
            is_possible = action.check_precond(kb, action.args)
            if is_possible:
                possible_actions.append(action)
        '''
        # A KB Abstract class holds a knowledge base of logical expressions, for propositional logic.

        # kb : class Propositional Knowledge Bases (PropKB)
        # kb.clauses : a list of all the sentences of the knowledge base.
        # kb.tell(self, sentence) : add a sentence to the clauses field of KB

        # decode_state : convert "TFFTFT" string of mapped positive and negative fluents to Expr object (FluentState object or state object)
        # FluentState.pos_sentence : list of positive expr

        kb = PropKB()
        positive_fluents = decode_state(state, self.state_map).pos_sentence()

        # add positive literals (sentence) to the clauses of KB
        kb.tell(positive_fluents)

        for action in self.actions_list:
            # Checks if the precondition is satisfied in the current state
            is_possible = True

            # Check for positive clauses
            for clause in action.precond_pos:
                if clause not in kb.clauses:
                    is_possible = False

            # Check for negative clauses
            for clause in action.precond_neg:
                if clause in kb.clauses:
                    is_possible = False

            if is_possible:
                possible_actions.append(action)

        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        # TODO implement
        new_state = FluentState([], [])

        old_state = decode_state(state, self.state_map)

        for pos_fluent in old_state.pos:
            if pos_fluent not in action.effect_rem: # Positive effect
                new_state.pos.append(pos_fluent)

        for pos_fluent in action.effect_add:       # Negative effect
            if pos_fluent not in new_state.pos:
                new_state.pos.append(pos_fluent)

        for neg_fluent in old_state.neg:
            if neg_fluent not in action.effect_add: # Negative effect
                new_state.neg.append(neg_fluent)

        for neg_fluent in action.effect_rem:       # Positive effect
            if neg_fluent not in new_state.neg:
                new_state.neg.append(neg_fluent)

        '''
        old_state = decode_state(state, self.state_map)
        pos, neg = set(old_state.pos), set(old_state.neg)
        pos = pos - set(action.effect_rem) | set(action.effect_add)
        neg = neg - set(action.effect_add) | set(action.effect_rem)
        new_state = FluentState(list(pos), list(neg))
        '''


        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        # kb : class Propositional Knowledge Bases (PropKB)
        # kb.clauses : a list of all the sentences of the knowledge base.
        # kb.tell(self, sentence) : add a sentence to the clauses field of KB

        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # Note that this is not a true heuristic
        h_const = 1
        return h_const

    def h_pg_levelsum(self, node: Node):
        '''
        This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        '''
        # Requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    def h_ignore_preconditions(self, node: Node):
        '''
        This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        '''
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)

        # kb : class Propositional Knowledge Bases (PropKB)
        # kb.clauses : a list of all the sentences of the knowledge base.
        # kb.tell(self, sentence) : add a sentence to the clauses field of KB

        # decode_state : convert "TFFTFT" string of mapped positive and negative fluents to Expr object (FluentState object or state object)

        # FluentState.pos_sentence : list of positive FluentState

        count = 0
        kb = PropKB()
        # convert all positive fluents into clauses of class Propositional Knowledge Bases (KB)
        kb.tell(decode_state(node.state, self.state_map).pos_sentence())

        # check if goals are found in clauses of KB
        for goal in self.goal:
            if goal not in kb.clauses:
                count += 1
        return count

        # 1
        '''
        kb = PropKB()
        # convert all positive fluents into clauses of class Propositional Knowledge Bases (KB)
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        return sum(c not in kb.clauses for c in self.goal)
        '''

        # 2
        '''
        goal = set(self.goal)
        actions = set(self.actions_list)
        # Included fluents so far
        fluents = set(decode_state(node.state, self.state_map).pos) & goal
        count = 0
        while fluents != goal:
            action = max(actions, key=lambda action: len((set(action.effect_add) - fluents) & goal))
            fluents = fluents | set(action.effect_add)
            actions = actions - set([action])
            count += 1
        return count
        '''

        # 3
        '''
        # self.state_map : list of positive and negative FluentState
        # node.state : state
        count = 0
        for goal in self.goal:
            for idx, s in enumerate(self.state_map):
                if s == goal & node.state[idx] = 'F':
                    count += 1
        return count
        '''

        # 4
        '''
        count = len(set(self.goal))
        positive_fluents = decode_state(state, self.state_map).pos_sentence()

        for precond in positive_fluents:
            if precond in self.goal:
                count -= 1

        return count
        '''

def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
           # C2
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           # C1
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           # P2
           expr('At(P1, JFK)'),
           # P2
           expr('At(P2, SFO)'),
           ]

    neg = create_expr_neg_list(pos, cargos, planes, airports)
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)

def air_cargo_p2() -> AirCargoProblem:
    # TODO implement Problem 2 definition
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)'),
           ]
            # C2
    neg = [expr('At(C2, SFO)'),
           expr('At(C2, ATL)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('In(C2, P3)'),
           # C1
           expr('At(C1, JFK)'),
           expr('At(C1, ATL)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('In(C1, P3)'),
           # C3
           expr('At(C3, JFK)'),
           expr('At(C3, SFO)'),
           expr('In(C3, P1)'),
           expr('In(C3, P2)'),
           expr('In(C3, P3)'),
           # P1
           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           # P2
           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           # P3
           expr('At(P3, SFO)'),
           expr('At(P3, JFK)'),
           ]

    #neg = create_expr_neg_list(pos, cargos, planes, airports)
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
    #pass


def air_cargo_p3() -> AirCargoProblem:
    # TODO implement Problem 3 definition
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']

    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
           # C2

    neg = [expr('At(C2, SFO)'),
           expr('At(C2, ATL)'),
           expr('At(C2, ORD)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           # C1
           expr('At(C1, JFK)'),
           expr('At(C1, ATL)'),
           expr('At(C1, ORD)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           # C3
           expr('At(C3, JFK)'),
           expr('At(C3, SFO)'),
           expr('At(C3, ORD)'),
           expr('In(C3, P1)'),
           expr('In(C3, P2)'),
           # C4
           expr('At(C4, JFK)'),
           expr('At(C4, SFO)'),
           expr('At(C4, ATL)'),
           expr('In(C4, P1)'),
           expr('In(C4, P2)'),
           # P1
           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           expr('At(P1, ORD)'),
           # P2
           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           expr('At(P2, ORD)'),
           ]

    #neg = create_expr_neg_list(pos, cargos, planes, airports)
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C3, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
    #pass


def create_expr_neg_list(expr_pos_list, cargos, planes, airports) -> list:
    s = set()
    for item in expr_pos_list:
        item = str(item)
        for cargo in cargos:
            for plane in planes:
                for airport in airports:
                    if cargo in item and airport not in item:
                        s.add(expr('At({},{})'.format(cargo, airport)))
                    if plane in item and airport not in item:
                        s.add(expr('At({},{})'.format(plane, airport)))
                    s.add(expr('In({},{})'.format(cargo, plane)))
    return list(s)
