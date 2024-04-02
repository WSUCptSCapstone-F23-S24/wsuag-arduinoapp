import math

def get_min_distance(plots_array, expected_coordinates):
    min_distance = float('inf')
    min_index = -1

    for i, plot in enumerate(plots_array):
        # distance between top left corner and expected corner
        curr_distance = math.sqrt((plot[0][0] - expected_coordinates[0])**2 + 
                                  (plot[0][1] - expected_coordinates[1])**2)
        if curr_distance < min_distance:
            min_distance = curr_distance
            min_index = i

    return plots_array[min_index], min_index

def distances_are_valid(plots_array, array_of_expected_coordinates, max_allowable_distance):
    used_indexes = []
    for curr in array_of_expected_coordinates:
        val, index = get_min_distance(plots_array, curr)
        if val[0] > max_allowable_distance:
            raise Exception("plot too far away from expected location")
        if index in used_indexes:
            raise Exception("two plots map to the same expected location")
        used_indexes.append(index)