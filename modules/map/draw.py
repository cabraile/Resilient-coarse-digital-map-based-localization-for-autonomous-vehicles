from matplotlib.pyplot import *
from matplotlib.patches import Ellipse

def draw_routes(routes, draw_endpoints=True, ax=None, color=None, suffix="", no_text=False):
    """
    @param routes: list of Route.
    @param draw_endpoint: bool. 
        Whether to scatter the endpoints from each way or not.
    @param ax: matplotlib.pyplot.Axis.
        Axis on which the graph is going to be drawn.
    @param color: NoneType, tuple or string.
        Color of the plot.
            If NoneType, color is set automatically; 
            if tuple, color is set as RGB (float).
            if string, color is defined by its name.
    @param suffix: string.
        Appended at the end of the label of each route.
    @return matplotlib.pyplot.Axis.
        The axis in which the operations were performed.
    """
    if(ax is None):
        ax = subplot(111)
    alpha = 1.
    if(color == "gray"):
        alpha=0.5
    colors_list = [ "blue", "#009900", "#6622AA","orange"]
    street_ids = []
    for route_idx, route in routes.items():
        route_idx = int(route_idx)
        # Line segments
        wpts = route.waypoints()
        xs = [p[0] for p in wpts]
        ys = [p[1] for p in wpts]
        color = colors_list[route_idx]
        ax.plot(xs, ys, label="Route {} {}".format(int(route_idx), suffix), color=color, alpha=alpha)

        # Line points
        if(draw_endpoints):
            ax.scatter(xs, ys, color=color)
        if(not no_text):
            # Street identifiers
            x_lims = ax.get_xlim()
            y_lims = ax.get_ylim()
            for way in route.get_way_list():
                street_id = way.street_id()
                p_init = way.p_init()
                p_end = way.p_end()
                center = ( (p_init[0] + p_end[0])/2. , (p_init[1] + p_end[1])/2. )
                if( (center[0] < x_lims[0] or center[0] > x_lims[1]) or
                    (center[1] < y_lims[0] or center[1] > y_lims[1]) or
                    street_id in street_ids
                ):
                    continue
                street_ids.append(street_id)
                pi = 3.1415
                angle = (way.orientation()/pi)*180
                if(angle >= 180):
                    angle = angle - 180
                if(angle < 0):
                    angle = angle + 180
                ax.text(center[0], center[1]+2, street_id, rotation=angle, color=color,alpha=alpha)
        
        # Landmarks
        lms = route.landmarks()
        xs = [ ]; ys = []
        for lm in lms:
            xs.append(lm.get_position()[0])
            ys.append(lm.get_position()[1])
        ax.scatter(xs,ys, marker="*", c="#FF9900", s=100)
    return ax

def draw_hypothesis(hypothesis, face_color, ax):
    """
    @brief Draw a hypothesis on the xy plane as an ellipsis.

    @param hypothesis: Hypothesis. 
        The hypothesis to be drawn.
    @param face_color: tuple.
        The color of the ellipsis of the hypothesis.
    @param ax: Matplotlib.pyplot.Axis.
        Axis on which the hypothesis is going to be drawn.
    @return matplotlib.pyplot.Axis.
        The axis in which the operations were performed.
    """
    route = hypothesis.route
    ret = route.from_distance_to_xy(hypothesis.mean)
    if(ret is None):
        return ax
    x,y = ret
    way,_ = route.get_by_distance(hypothesis.mean)
    angle = way.orientation()
    confidence_interval = 2.0 * (hypothesis.variance ** 0.5)
    ellipse = Ellipse(
        xy=(x,y), height=confidence_interval, width = confidence_interval, angle=angle
    )
    ax.scatter([x],[y], s=100, marker="P",c="k")
    ax.add_artist(ellipse)
    ellipse.set_facecolor(face_color)
    ellipse.set_edgecolor((1,0,0))
    return ax

def draw_hypotheses(hypotheses, face_color=(0,0,0,0), ax=None):
    """
    @brief Draw hypotheses as ellipses on the xy pĺane. 
    The most likely hypothesis is represented by a green face ellipsis.

    @param hypotheses: list of Hypothesis. 
        The hypothesis to be drawn.
    @param face_color: tuple.
        The color of the ellipsis of the hypothesis.
    @param ax: Matplotlib.pyplot.Axis.
        Axis on which the hypothesis is going to be drawn.
    @return matplotlib.pyplot.Axis.
        The axis in which the operations were performed.
    """
    if(ax is None):
        ax = subplot(111)

    best_hypothesis = hypotheses.get_best_hypothesis()

    # Draw all hypotheses but the most likely one
    for hypothesis in hypotheses.get().values():
        if(hypothesis.route.idx != best_hypothesis.route.idx):
            ax = draw_hypothesis(hypothesis, face_color, ax)
    
    # Draw the most likely hypothesis
    ax = draw_hypothesis(best_hypothesis, face_color=(0,1,0,0.5), ax=ax)
    return ax

def zoom_to(x,y , dist, ax):
    """
    @brief zoom the ax plot to a given xy position.

    @param x: float.
        x position where the plot is centered.
    @param y: float.
        y position where the plot is centered.
    @param dist: float.
        Distance from which the limits are going to be from the center.
    @param ax: Matplotlib.pyplot.Axis.
        Axis on which the zoom will be performed.
    @return matplotlib.pyplot.Axis.
        The axis in which the operations were performed.
    """
    ax.set_xlim([x-dist, x+dist])
    ax.set_ylim([y-dist, y+dist])
    return ax

def mark_point_from_route(distance, route, ax=None):
    """
    @brief Draw a point in the xy graph given its route bounderies.

    @param distance: float.
        The distance the point is on the route
    @param route: Route.
        The route in which the point is contained.
    @param ax: Matplotlib.pyplot.Axis.
        Axis on which the point is going to be drawn.
    @return matplotlib.pyplot.Axis.
        The axis in which the operations were performed.
    """
    ret = route.from_distance_to_xy(distance)
    if(ret is None):
        return ax
    x, y = ret
    ax.scatter([x], [y], marker="x",color="magenta", s=100)
    return ax
