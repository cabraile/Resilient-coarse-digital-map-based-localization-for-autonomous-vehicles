from modules.map.draw import draw_routes, draw_hypotheses
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox, OffsetBox

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, zoomed_inset_axes

def draw_figure_routes(
    plt_axes, 
    hypotheses, 
    groundtruth_xy, 
    active_routes, 
    inactive_routes, 
    current_image, 
    detected_landmarks,
    zoom
):
    plt_axes["global_scope"].cla(); plt_axes["global_scope"].set_title("Complete route")
    draw_routes(inactive_routes, draw_endpoints=False, color="gray", ax=plt_axes["global_scope"], suffix="(inactive)", no_text=True)
    draw_routes(active_routes, draw_endpoints=True, ax=plt_axes["global_scope"], suffix="(active)", no_text=True)
    draw_hypotheses(hypotheses, ax=plt_axes["global_scope"])
    plt_axes["global_scope"].scatter( [groundtruth_xy[0]], [groundtruth_xy[1]], marker="x", color="red", s=100)
    plt_axes["global_scope"].legend(loc="upper center")
    plt_axes["global_scope"].axis("off")

    # Zoomed
    axins = inset_axes(plt_axes["global_scope"], loc='lower left', height="30%", width="30%")
    draw_routes(inactive_routes, draw_endpoints=False, color="gray", ax=axins, suffix="(inactive)", no_text=True)
    draw_routes(active_routes, draw_endpoints=True, ax=axins, suffix="(active)", no_text=True)
    draw_hypotheses(hypotheses, ax=axins)
    axins.scatter([groundtruth_xy[0]], [groundtruth_xy[1]], marker="x", color="red", s=100)
    axins.axes.xaxis.set_visible(False)
    axins.axes.yaxis.set_visible(False)
    axins.set_xlim(groundtruth_xy[0]-zoom, groundtruth_xy[0]+zoom)
    axins.set_ylim(groundtruth_xy[1]-zoom, groundtruth_xy[1]+zoom)
    mark_inset(plt_axes["global_scope"], axins, loc1=1, loc2=2, fc="none", ec="0.5")

    # Current Frame
    axins_camera = inset_axes(plt_axes["global_scope"], loc='upper right', height="40%", width="40%")  # zoom = 1.5
    axins_camera.axes.xaxis.set_visible(False)
    axins_camera.axes.yaxis.set_visible(False)
    axins_camera.imshow(current_image)

    # Landmark detections
    for landmark in detected_landmarks:
        p = landmark.get_position()
        img = landmark.images()[0]
        imagebox = OffsetImage(img, zoom=0.005)
        ab1 = AnnotationBbox(imagebox, p, xycoords='data', pad=0, frameon=False)
        plt_axes["global_scope"].add_artist(ab1)
        imagebox = OffsetImage(img, zoom=0.005)
        ab2 = AnnotationBbox(imagebox, p, xycoords='data', pad=0, frameon=False)
        axins.add_artist(ab2)
    return 

def draw_landmarks(plt_axes, query_img, landmark_img):
    plt_axes["last_match"].set_title("Last match")
    axins_landmark = inset_axes(plt_axes["last_match"], loc='lower right', height="50%", width="50%")  # zoom = 1.5
    plt_axes["last_match"].imshow(query_img)
    axins_landmark.imshow(landmark_img)
    plt_axes["last_match"].axis("off")
    axins_landmark.axis("off")
    return

def save_figure_map(plt_fig, plt_axes, ignore=False, label=""):
    if(ignore):
        return
    extent = plt_axes["global_scope"].get_window_extent().transformed( plt_fig.dpi_scale_trans.inverted() )
    plt_fig.savefig('/home/andre/Desktop/imgs_paper/{}_{}_{}.png'.format(step, "map", label), bbox_inches = extent.expanded(1.1, 1.2) )
    return

def save_figure_landmark(plt_fig, plt_axes, ignore=False):
    if(ignore):
        return
    extent = plt_axes["last_match"].get_window_extent().transformed( plt_fig.dpi_scale_trans.inverted() )
    plt_fig.savefig('/home/andre/Desktop/imgs_paper/{}_{}.png'.format(step, "landmark"), bbox_inches = extent.expanded(1.1, 1.2) )
    return
