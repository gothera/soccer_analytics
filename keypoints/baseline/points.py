def scale_points(points_dict, s_width, s_height):
    """
    Scale points by s_width and s_height factors
    :param points_dict: dictionary of annotations/predictions with normalized point values
    :param s_width: width scaling factor
    :param s_height: height scaling factor
    :return: dictionary with scaled points
    """
    line_dict = {}
    
    for line_class, points in points_dict.items():
        scaled_points = []
        for point in points:
            new_point = {'x': point['x'] * (s_width-1), 'y': point['y'] * (s_height-1)}
            scaled_points.append(new_point)
        if len(scaled_points):
            line_dict[line_class] = scaled_points
    return line_dict