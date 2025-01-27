import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc

class BasketballCourt:
    """Static class variables specified by NBA rules"""
    COURT_LENGTH = 28.6512  # meters
    COURT_WIDTH = 15.24   # meters
    BASKET_HEIGHT = 3.05  # meters (10 feet)
    BASKET_FROM_BASELINE = 1.2192  # meters (4 feet)
    FREE_THROW_LINE_DIST = 4.572  # meters (15 feet)
    CENTER_CIRCLE_RADIUS = 1.8288  # meters (6 feet)
    THREE_POINT_LINE_RADIUS = 7.239  # meters (23'9")
    THREE_POINT_LINE_CORNER_DIST = 6.7056  # meters (22 feet)
    KEY_WIDTH = 4.8768  # meters (16 feet)
    KEY_LENGTH = 5.7912  # meters (19 feet)
    BACKBOARD_WIDTH = 1.8288  # meters (6 feet)
    RIM_DIAMETER = 0.4572  # meters (18 inches)
    THREE_POINT_CORNER_LENGTH = 4.2672
    LINE_WIDTH = 0.0508
    # THREE_POINT_CORNER_WIDTH = 0.914
    
    def __init__(self):
        """Initialize 3D coordinates of all elements of the basketball court"""
        # Center of court
        self.center_mark = np.array([0, 0, 0], dtype='float')
        
        # Court corners
        self.top_left_corner = np.array([-self.COURT_LENGTH/2, -self.COURT_WIDTH/2, 0], dtype='float')
        self.top_right_corner = np.array([self.COURT_LENGTH/2, -self.COURT_WIDTH/2, 0], dtype='float')
        self.bottom_left_corner = np.array([-self.COURT_LENGTH/2, self.COURT_WIDTH/2, 0], dtype='float')
        self.bottom_right_corner = np.array([self.COURT_LENGTH/2, self.COURT_WIDTH/2, 0], dtype='float')
        
        # Center circle marks
        self.center_circle_top = np.array([0, -self.CENTER_CIRCLE_RADIUS, 0], dtype='float')
        self.center_circle_bottom = np.array([0, self.CENTER_CIRCLE_RADIUS, 0], dtype='float')
        
        # Free throw lines
        self.left_free_throw_line_center = np.array([-self.COURT_LENGTH/2 + self. BASKET_FROM_BASELINE + self.FREE_THROW_LINE_DIST, 0, 0], dtype='float')
        self.right_free_throw_line_center = np.array([self.COURT_LENGTH/2 - self.BASKET_FROM_BASELINE - self.FREE_THROW_LINE_DIST , 0, 0], dtype='float')
        
        # Key (painted area) corners - Left
        self.left_key_top_left = np.array([-self.COURT_LENGTH/2, -self.KEY_WIDTH/2, 0], dtype='float')
        self.left_key_top_right = np.array([-self.COURT_LENGTH/2 + self.KEY_LENGTH, -self.KEY_WIDTH/2, 0], dtype='float')
        self.left_key_bottom_left = np.array([-self.COURT_LENGTH/2, self.KEY_WIDTH/2, 0], dtype='float')
        self.left_key_bottom_right = np.array([-self.COURT_LENGTH/2 + self.KEY_LENGTH, self.KEY_WIDTH/2, 0], dtype='float')
        
        # Key (painted area) corners - Right
        self.right_key_top_left = np.array([self.COURT_LENGTH/2 - self.KEY_LENGTH, -self.KEY_WIDTH/2, 0], dtype='float')
        self.right_key_top_right = np.array([self.COURT_LENGTH/2, -self.KEY_WIDTH/2, 0], dtype='float')
        self.right_key_bottom_left = np.array([self.COURT_LENGTH/2 - self.KEY_LENGTH, self.KEY_WIDTH/2, 0], dtype='float')
        self.right_key_bottom_right = np.array([self.COURT_LENGTH/2, self.KEY_WIDTH/2, 0], dtype='float')
        
        # Three point line intersection points
        self.left_three_point_corner_top = np.array([-self.COURT_LENGTH/2 + self.THREE_POINT_LINE_CORNER_DIST, -self.COURT_WIDTH/2, 0], dtype='float')
        self.left_three_point_corner_bottom = np.array([-self.COURT_LENGTH/2 + self.THREE_POINT_LINE_CORNER_DIST, self.COURT_WIDTH/2, 0], dtype='float')
        self.right_three_point_corner_top = np.array([self.COURT_LENGTH/2 - self.THREE_POINT_LINE_CORNER_DIST, -self.COURT_WIDTH/2, 0], dtype='float')
        self.right_three_point_corner_bottom = np.array([self.COURT_LENGTH/2 - self.THREE_POINT_LINE_CORNER_DIST, self.COURT_WIDTH/2, 0], dtype='float')
        
        # Basket positions (including height)
        self.left_basket = np.array([-self.COURT_LENGTH/2 + self.BASKET_FROM_BASELINE, 0, self.BASKET_HEIGHT], dtype='float')
        self.right_basket = np.array([self.COURT_LENGTH/2 - self.BASKET_FROM_BASELINE, 0, self.BASKET_HEIGHT], dtype='float')
        
        # Backboard edges
        self.left_backboard_top = np.array([-self.COURT_LENGTH/2 + self.BASKET_FROM_BASELINE, -self.BACKBOARD_WIDTH/2, self.BASKET_HEIGHT + 0.6], dtype='float')
        self.left_backboard_bottom = np.array([-self.COURT_LENGTH/2 + self.BASKET_FROM_BASELINE, self.BACKBOARD_WIDTH/2, self.BASKET_HEIGHT + 0.6], dtype='float')
        self.right_backboard_top = np.array([self.COURT_LENGTH/2 - self.BASKET_FROM_BASELINE, -self.BACKBOARD_WIDTH/2, self.BASKET_HEIGHT + 0.6], dtype='float')
        self.right_backboard_bottom = np.array([self.COURT_LENGTH/2 - self.BASKET_FROM_BASELINE, self.BACKBOARD_WIDTH/2, self.BASKET_HEIGHT + 0.6], dtype='float')
        
        # Dictionary for easy access to points
        self.point_dict = {
            "CENTER_MARK": self.center_mark,
            "TL_CORNER": self.top_left_corner,
            "TR_CORNER": self.top_right_corner,
            "BL_CORNER": self.bottom_left_corner,
            "BR_CORNER": self.bottom_right_corner,
            "LEFT_BASKET": self.left_basket,
            "RIGHT_BASKET": self.right_basket,
            "LEFT_FREE_THROW": self.left_free_throw_line_center,
            "RIGHT_FREE_THROW": self.right_free_throw_line_center,
            "LEFT_KEY_TL": self.left_key_top_left,
            "LEFT_KEY_TR": self.left_key_top_right,
            "LEFT_KEY_BL": self.left_key_bottom_left,
            "LEFT_KEY_BR": self.left_key_bottom_right,
            "RIGHT_KEY_TL": self.right_key_top_left,
            "RIGHT_KEY_TR": self.right_key_top_right,
            "RIGHT_KEY_BL": self.right_key_bottom_left,
            "RIGHT_KEY_BR": self.right_key_bottom_right,
            "LEFT_THREE_POINT_TOP": self.left_three_point_corner_top,
            "LEFT_THREE_POINT_BOTTOM": self.left_three_point_corner_bottom,
            "RIGHT_THREE_POINT_TOP": self.right_three_point_corner_top,
            "RIGHT_THREE_POINT_BOTTOM": self.right_three_point_corner_bottom,
            "LEFT_BACKBOARD_TOP": self.left_backboard_top,
            "LEFT_BACKBOARD_BOTTOM": self.left_backboard_bottom,
            "RIGHT_BACKBOARD_TOP": self.right_backboard_top,
            "RIGHT_BACKBOARD_BOTTOM": self.right_backboard_bottom
        }
    
    def points(self):
        """Return list of all court points"""
        return list(self.point_dict.values())

def draw_court(court):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw outer court lines
    court_points = [
        court.top_left_corner,
        court.top_right_corner,
        court.bottom_right_corner,
        court.bottom_left_corner,
        court.top_left_corner
    ]
    points = np.array(court_points)
    ax.plot(points[:, 0], points[:, 1], 'b-')
    
    # Draw center line
    ax.plot([-0.1, 0.1], [court.COURT_WIDTH/2, court.COURT_WIDTH/2], 'b-')
    ax.plot([-0.1, 0.1], [-court.COURT_WIDTH/2, -court.COURT_WIDTH/2], 'b-')
    
    # Draw center circle
    center_circle = Circle((0, 0), court.CENTER_CIRCLE_RADIUS, fill=False, color='b')
    ax.add_patch(center_circle)
    
    # Left key
    left_key = [
        court.left_key_top_left,
        court.left_key_top_right,
        court.left_key_bottom_right,
        court.left_key_bottom_left,
        court.left_key_top_left
    ]
    points = np.array(left_key)
    ax.plot(points[:, 0], points[:, 1], 'b-')
    
    # Right key
    right_key = [
        court.right_key_top_left,
        court.right_key_top_right,
        court.right_key_bottom_right,
        court.right_key_bottom_left,
        court.right_key_top_left
    ]
    points = np.array(right_key)
    ax.plot(points[:, 0], points[:, 1], 'b-')
    
    # Draw free throw circles
    left_ft_circle = Circle(
        (court.left_free_throw_line_center[0], court.left_free_throw_line_center[1]),
        court.CENTER_CIRCLE_RADIUS,
        fill=False,
        color='b'
    )
    right_ft_circle = Circle(
        (court.right_free_throw_line_center[0], court.right_free_throw_line_center[1]),
        court.CENTER_CIRCLE_RADIUS,
        fill=False,
        color='b'
    )
    ax.add_patch(left_ft_circle)
    ax.add_patch(right_ft_circle)
    
    # Draw three-point lines
    # Draw three-point lines
    left_three_point = Arc(
        (-court.COURT_LENGTH/2 + court.BASKET_FROM_BASELINE + court.RIM_DIAMETER/2 + court.LINE_WIDTH, 0), 
        2 * court.THREE_POINT_LINE_RADIUS,
        2 * court.THREE_POINT_LINE_RADIUS,
        theta1=-np.degrees(np.arccos((court.THREE_POINT_CORNER_LENGTH - court.BASKET_FROM_BASELINE - court.RIM_DIAMETER/2) / court.THREE_POINT_LINE_RADIUS)),
        theta2=np.degrees(np.arccos((court.THREE_POINT_CORNER_LENGTH - court.BASKET_FROM_BASELINE - court.RIM_DIAMETER/2) / court.THREE_POINT_LINE_RADIUS)),
        color='b'
    )
    right_three_point = Arc(
        (court.COURT_LENGTH/2 - court.BASKET_FROM_BASELINE - court.RIM_DIAMETER/2, 0),
        2 * court.THREE_POINT_LINE_RADIUS,
        2 * court.THREE_POINT_LINE_RADIUS,
        theta1=180-np.degrees(np.arccos((court.THREE_POINT_CORNER_LENGTH - court.BASKET_FROM_BASELINE - court.RIM_DIAMETER/2) / court.THREE_POINT_LINE_RADIUS)),
        theta2=180+np.degrees(np.arccos((court.THREE_POINT_CORNER_LENGTH - court.BASKET_FROM_BASELINE - court.RIM_DIAMETER/2) / court.THREE_POINT_LINE_RADIUS)),
        color='b'
    )
    ax.add_patch(left_three_point)
    ax.add_patch(right_three_point)
    
    ax.plot([-court.COURT_LENGTH/2, -court.COURT_LENGTH/2 + court.THREE_POINT_CORNER_LENGTH],
            [-court.THREE_POINT_LINE_CORNER_DIST, -court.THREE_POINT_LINE_CORNER_DIST], 'b-')
    ax.plot([-court.COURT_LENGTH/2, -court.COURT_LENGTH/2 + court.THREE_POINT_CORNER_LENGTH],
            [court.THREE_POINT_LINE_CORNER_DIST, court.THREE_POINT_LINE_CORNER_DIST], 'b-')
    ax.plot([court.COURT_LENGTH/2, court.COURT_LENGTH/2 - court.THREE_POINT_CORNER_LENGTH],
            [-court.THREE_POINT_LINE_CORNER_DIST, -court.THREE_POINT_LINE_CORNER_DIST], 'b-')
    ax.plot([court.COURT_LENGTH/2, court.COURT_LENGTH/2 - court.THREE_POINT_CORNER_LENGTH],
            [court.THREE_POINT_LINE_CORNER_DIST, court.THREE_POINT_LINE_CORNER_DIST], 'b-')
    
    # Draw baskets
    left_basket = Circle(
        (court.left_basket[0], court.left_basket[1]),
        court.RIM_DIAMETER/2,
        fill=False,
        color='r'
    )
    right_basket = Circle(
        (court.right_basket[0], court.right_basket[1]),
        court.RIM_DIAMETER/2,
        fill=False, 
        color='r'
    )
    ax.add_patch(left_basket)
    ax.add_patch(right_basket)
    
    # Set aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-court.COURT_LENGTH/2 - 1, court.COURT_LENGTH/2 + 1)
    ax.set_ylim(-court.COURT_WIDTH/2 - 1, court.COURT_WIDTH/2 + 1)
    plt.axis('off')
    plt.savefig('./basket_pitch.png', format='png', bbox_inches='tight')
    
    return fig, ax


INTERSECTION_TO_PITCH_POINTS = {
    0: 'TL_CORNER',
    1: 'TR_CORNER', 
    2: 'BL_CORNER',
    3: 'BR_CORNER',
    4: 'CENTER_MARK',
    5: 'LEFT_BASKET',
    6: 'RIGHT_BASKET',
    7: 'LEFT_FREE_THROW',
    8: 'RIGHT_FREE_THROW',
    9: 'LEFT_KEY_TL',
    10: 'LEFT_KEY_TR',
    11: 'LEFT_KEY_BL',
    12: 'LEFT_KEY_BR',
    13: 'RIGHT_KEY_TL',
    14: 'RIGHT_KEY_TR',
    15: 'RIGHT_KEY_BL',
    16: 'RIGHT_KEY_BR',
    17: 'LEFT_THREE_POINT_TOP',
    18: 'LEFT_THREE_POINT_BOTTOM',
    19: 'RIGHT_THREE_POINT_TOP',
    20: 'RIGHT_THREE_POINT_BOTTOM',
    21: 'LEFT_BACKBOARD_TOP',
    22: 'LEFT_BACKBOARD_BOTTOM',
    23: 'RIGHT_BACKBOARD_TOP',
    24: 'RIGHT_BACKBOARD_BOTTOM',
    25: 'CENTER_CIRCLE_TOP',
    26: 'CENTER_CIRCLE_BOTTOM',
    27: 'LEFT_KEY_FREE_THROW_TOP',
    28: 'LEFT_KEY_FREE_THROW_BOTTOM',
    29: 'RIGHT_KEY_FREE_THROW_TOP',
    30: 'RIGHT_KEY_FREE_THROW_BOTTOM',
    31: 'LEFT_THREE_POINT_ARC_TOP',
    32: 'LEFT_THREE_POINT_ARC_BOTTOM',
    33: 'RIGHT_THREE_POINT_ARC_TOP',
    34: 'RIGHT_THREE_POINT_ARC_BOTTOM'
}

PITCH_POINTS_TO_INTERSECTON = {v: k for k, v
                               in INTERSECTION_TO_PITCH_POINTS.items()}

def plot_pitch_points(data_dict, bottom_left, top_right, output_image_path=None):
    """
    Plots [x, y] points on a 2D plane from a dictionary where each key maps to a single [x, y, z] point.
    Only points where z == 0 and within the defined rectangular pitch are plotted.
    The plot is resized to 5120x2880 pixels and can be saved as an image.

    Parameters:
    - data_dict (dict): Dictionary with string keys and values as [x, y, z] points.
    - bottom_left (list or tuple): [x, y] coordinates of the bottom-left corner of the pitch.
    - top_right (list or tuple): [x, y] coordinates of the top-right corner of the pitch.
    - output_image_path (str, optional): File path to save the plot image. If None, the plot is displayed.

    Returns:
    - None
    """
    # Extract pitch boundaries
    pitch_x_min, pitch_y_min = bottom_left
    pitch_x_max, pitch_y_max = top_right

    # Prepare lists to hold points within the pitch
    plot_points = {}
    
    for key, point in data_dict.items():
        x, y = point
        plot_points[key] = (x, y)

    if not plot_points:
        print("No points to plot within the specified pitch and z=0.")
        return

        # Configure figure size and DPI for 5120x2880 pixels
    # Figure size in inches = pixels / DPI
    desired_width_px = 5120
    desired_height_px = 2880
    dpi = 320  # 5120 / 16 = 320, 2880 / 9 = 320 for 16:9 ratio
    fig_width_in = desired_width_px / dpi  # 16 inches
    fig_height_in = desired_height_px / dpi  # 9 inches

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)

    # Set the axes limits to match the pitch boundaries exactly
    ax.set_xlim(pitch_x_min, pitch_x_max)
    ax.set_ylim(pitch_y_min, pitch_y_max)

    # Draw the rectangular pitch
    pitch_width = pitch_x_max - pitch_x_min
    pitch_height = pitch_y_max - pitch_y_min
    pitch_rectangle = plt.Rectangle(
        (pitch_x_min, pitch_y_min),
        pitch_width,
        pitch_height,
        linewidth=2,
        edgecolor='black',
        facecolor='none'
    )
    ax.add_patch(pitch_rectangle)

    # Plot each point
    for key, (x, y) in plot_points.items():
        ax.scatter(x, y, color='red', s=50)  # Customize point color and size as needed
        ax.text(x+0.5, y+0.5, str(PITCH_POINTS_TO_INTERSECTON[key]))
    
    lines = {
        0:1,
        2:3,
        9:10,
        11:12,
        10:12,
        13:14,
        15:16,

    }

    for k, v in lines.items():
        p1, p2 = INTERSECTION_TO_PITCH_POINTS[k], INTERSECTION_TO_PITCH_POINTS[v]
        xs = plot_points[p1][0], plot_points[p2][0]
        ys = plot_points[p1][1], plot_points[p2][1]
        ax.plot(xs, ys, color='b')

    # Remove all axes, ticks, labels, and spines
    ax.axis('off')

    if output_image_path:
        # Ensure the directory exists
        plt.savefig(output_image_path, format='png', dpi=dpi, bbox_inches='tight')
        print(f"Plot saved as {output_image_path}")
    plt.show()
    plt.close()

def filter_2d_points(data_dict):
    ret_dict = {}
    # Iterate over each key-value pair in the dictionary
    for key, point in data_dict.items():
        x, y, z = point
        if z == 0.:
            ret_dict[key] = [x, y]
    return ret_dict

if __name__ == "__main__":
    court = BasketballCourt()
    fig, ax = draw_court(court)
    plt.show()