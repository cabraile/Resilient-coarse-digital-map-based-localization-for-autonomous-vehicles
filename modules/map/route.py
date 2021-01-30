from .way import Way
from numpy import cos, sin, array, argsort, sqrt, inf
from numpy.linalg import norm
import yaml
import utm
from .landmark import Landmark

class Route():

    def __init__(self):
        self.__ways__               = []    # List of ways - not necessarily organized by their seq ids or unique ids
        self.__ways_seq_ids__       = {}    # Maps the sequential id of each way to their index on self.ways list
        self.__ways_street_ids__    = {}    # Maps the street id to the indices of ways that belong to that street segment
        self.__cumulative_length__  = []    # Given the seq_id, retrieves the cumulative distance from the begin of the route to the initial point of the way
        self.__n_ways__ = 0                 # The number of ways in the route
        self.__total_length__ = 0.0         # Total distance driven along this route
        self.__waypoints__ = []             # The list of points (x,y tuples) that belong on each way (either begin or end). Not indexed.
        self.__domain__ = (0, 0)            # The minimum and maximum length
        self.__landmarks__ = {              # The loaded landmarks for this route
            "range" : [],                   # - The list of ranges from/to along the route where the landmark can be detected
            "object" : [],                  # - The list of Landmark instances
            "count" : 0                     # - The number of landmarks in that route
        }             
        self.idx = None                # Sets an index to the route - used for simulation
        return

    def from_yaml(self, path):
        route_dict = None
        with open(path) as f:
            route_dict = yaml.load(f, Loader=yaml.FullLoader)

        # Load ways
        n_ways = len(route_dict["ways"])
        cumulative_length = 0
        for seq_id in range(n_ways):
            way_entry = route_dict["ways"][seq_id]
            way_uid = way_entry["uid"]
            street_id = way_entry["street_id"]
            way = Way(
                way_uid = way_uid,
                street_id = street_id,
                coord_init = way_entry["p_init"], 
                coord_end = way_entry["p_end"], 
                maxspeed = int( way_entry["maxspeed"] ), 
                oneway = way_entry["oneway"],
            )
            list_id = len(self.__ways__)
            self.__ways_seq_ids__[seq_id] = list_id
            if(street_id not in self.__ways_street_ids__):
                self.__ways_street_ids__[street_id] = [] 
            self.__ways_street_ids__[street_id].append(list_id)
            self.__cumulative_length__.append(cumulative_length)
            if(seq_id == 0):
                self.__waypoints__.append(way.p_init())
            self.__waypoints__.append(way.p_end())
            self.__ways__.append(way)
            cumulative_length += way.length()

        self.__total_length__ = cumulative_length
        self.__domain__ = (0, self.__total_length__)
        self.__n_ways__ = n_ways

        # Load landmarks
        count = 0
        for lm_path in route_dict["landmarks"]:
            lm = Landmark()
            lm.from_yaml(lm_path)
            self.__landmarks__["range"].append(self.__range_from_landmark__(lm,3.5))
            self.__landmarks__["object"].append(lm)
            count += 1
        self.__landmarks__["count"] = count
        return

    def from_data(self, ways, landmarks):
        """
        @param ways: dict. Dict of ways in which each key is the sequence index;
        @param landmarks: dict. Dict of landmarks computed for the waypoints provided. Indexed by their osmid.
        """
        cumulative_length = 0
        for seq_id, way in ways.items():
            list_id = len(self.__ways__)
            self.__ways_seq_ids__[seq_id] = list_id
            if( way.street_id() not in self.__ways_street_ids__ ):
                self.__ways_street_ids__[way.street_id()] = [] 
            self.__ways_street_ids__[way.street_id()].append(list_id)
            self.__cumulative_length__.append(cumulative_length)
            if( seq_id == 0 ):
                self.__waypoints__.append( way.p_init() )
            self.__waypoints__.append( way.p_end() )
            self.__ways__.append( way )
            cumulative_length += way.length()

        self.__total_length__ = cumulative_length
        self.__domain__ = (0, self.__total_length__)
        self.__n_ways__ = len(ways)

        # Load landmarks
        count = 0
        for lm in landmarks.values():
            self.__landmarks__["range"].append(self.__range_from_landmark__(lm,2.0))
            self.__landmarks__["object"].append(lm)
            count += 1
        self.__landmarks__["count"] = count
        return

    def __range_from_landmark__(self, lm, radius):
        L = array(lm.get_position())
        n_ways = self.__n_ways__
        # The distance from the landmark to the closest point defined by the way
        D = [inf for i in range(n_ways)]
        for i in range(n_ways):
            W = self.__ways__[i]
            W_prev = array([W.p_init()[0],W.p_init()[1]])
            W_curr = array([W.p_end()[0], W.p_end()[1]])
            V_1 = L - W_curr
            V_2 = W_curr - W_prev
            if  norm(V_1) * norm(V_2) == 0:
                D[i] = inf
                continue
            sin_alpha_2 = 1.0 - ( ( V_1.dot(V_2) ) / ( norm(V_1) * norm(V_2) ) )
            D[i] =  sqrt(norm(V_1) / (sin_alpha_2))

        # Project landmark on the route
        w_ids = argsort(D)
        W_min = self.__ways__[w_ids[0]]
        W_1 = array([W_min.p_init()[0], W_min.p_init()[1]])
        W_2 = array([W_min.p_end()[0], W_min.p_end()[1]])
        slope = (W_2[1] - W_1[1])/ (W_2[0] - W_1[0])
        intercept = W_1[1] - slope * W_1[0]
        perp_slope = -1./slope
        perp_intercept = L[1] - perp_slope * L[0]
        P_n_x = (perp_intercept - intercept)/(slope - perp_slope)
        P_n_y = slope * P_n_x + intercept
        P_n = array([P_n_x, P_n_y])
        # Excelente conversor:  https://sigam.ambiente.sp.gov.br/sigam3/Controles/latlongutm.htm?latTxt=ctl00_conteudo_TabNavegacao_TBCadastro_carCadastro_numLatitude&lonTxt=ctl00_conteudo_TabNavegacao_TBCadastro_carCadastro_numLongitude

        # Detection range to the landmark
        detection_range = [0,0]
        detection_range[0] = self.__cumulative_length__[w_ids[0]] + norm(P_n - W_1) - radius
        detection_range[1] = self.__cumulative_length__[w_ids[0]] + norm(P_n - W_1)
        return detection_range

    def get_landmark_range(self, lm_id):
        for i in range(self.__landmarks__["count"]):
            lm = self.__landmarks__["object"][i]
            if(lm.id == lm_id):
                return self.__landmarks__["range"][i]
        return None

    def detectable_landmarks_at(self, distance):
        lms = []
        for idx in range(self.__landmarks__["count"]):
            lm_range = self.__landmarks__["range"][idx]
            if(distance >= lm_range[0]) and (distance <= lm_range[1]):
                lm = self.__landmarks__["object"][idx]
                lms.append(lm)
        return lms

    def landmarks(self):
        return self.__landmarks__["object"][:]

    def domain(self):
        return self.__domain__

    def total_length(self):
        return self.__total_length__

    def get_by_distance(self, distance):
        """
        Retrieves a way given the distance driven on the route.
        """
        for seq_id in range(self.__n_ways__ - 1):
            if ( 
                (distance >= self.__cumulative_length__[seq_id]) and 
                (distance <= self.__cumulative_length__[seq_id+1])
            ):
                return self.__ways__[ self.__ways_seq_ids__[seq_id] ], seq_id
        if  distance <= self.__total_length__:
            return self.__ways__[ self.__ways_seq_ids__[self.__n_ways__-1] ], self.__n_ways__ - 1
        return None, None

    def from_distance_to_xy(self, distance):
        way, seq_id = self.get_by_distance(distance)
        if(way is None):
            return None
        angle = way.orientation()
        p_init = way.p_init()
        offset = distance - self.__cumulative_length__[seq_id]
        x = p_init[0] + offset * cos(angle)
        y = p_init[1] + offset * sin(angle)
        return (x,y)

    def get_way_list(self):
        return self.__ways__[:]

    def waypoints(self):
        return self.__waypoints__[:]

    def get_by_seq_id(self, id):
        """
        Retrieve the way given its sequential id on the route
        """
        if(id not in self.__ways_seq_ids__):
            return None
        return self.__ways__[ self.__ways_seq_ids__[id] ]

    def get_by_street_id(self, id):
        """
        Retrieve the list of ways that belong to the street of identifier = id
        """
        ways_list = []
        for id in self.__ways_street_ids__[id]:
            ways_list.append(self.__ways__[id])
        return ways_list