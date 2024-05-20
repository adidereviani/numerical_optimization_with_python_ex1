import matplotlib.pyplot as plt
from tests.examples import *


def plot_function_values(dict_a, dict_b, title=None):
    dict_list = [dict_a, dict_b]
    plt.figure()
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Function values', fontsize=12)
    plt.title(f'Comparison of Gradient and Newton Methods: {title} function', fontsize=18)
    length_difference = abs(len(dict_a['Function value']) - len(dict_b['Function value'])) * 0.5
    ratio_of_lengths = max(len(dict_a['Function value']), len(dict_b['Function value'])) / min(len(dict_a['Function value']), len(dict_b['Function value']))
    plt.scatter(x=[1], y=[dict_a['Function value'][0]], linewidth=4, color="red")
    for dictionary in dict_list:
        x_values = np.arange(1, len(dictionary['Function value']) + 1)
        y_values = dictionary['Function value']
        plt.plot(x_values, y_values, label=dictionary['Minimization class'])
    if ratio_of_lengths > 100:
        plt.xscale('log')
        plt.xlim(left=1)
    else:
        plt.xticks(np.arange(0, length_difference + 1, 5))
        plt.xlim(left=1, right=length_difference)
    plt.legend()
    plt.show()


def plot_contour_2d(f, dict_a, dict_b, title=None):
    dict_list = [dict_a, dict_b]
    location = []

    for i in dict_list:
        for col in range(i['Function location'].shape[1]):
            location.append(np.linalg.norm(i['Function location'][:, col]))
    maximum_location_norm = max(location)
    x_y = np.linspace(-maximum_location_norm, maximum_location_norm, 40)

    x_coordinates, y_coordinates = np.meshgrid(x_y, x_y)
    z = f.evaluate_function(np.array([np.array(x_coordinates).reshape(-1), np.array(y_coordinates).reshape(-1)]))
    z_coordinates = z.reshape(x_coordinates.shape)
    x1_dict_a, x2_dict_a, z_dict_a = dict_a['Function location'][0], dict_a['Function location'][1],  dict_a['Function value']
    x1_dict_b, x2_dict_b, z_dict_b = dict_b['Function location'][0], dict_b['Function location'][1], dict_b['Function value']
    plt.figure(figsize=(14, 9))
    plt.plot(x1_dict_a, x2_dict_a, label='Gradient descent')
    plt.plot(x1_dict_b, x2_dict_b, label='Newton descent')
    plt.contour(x_coordinates, y_coordinates, z_coordinates, 80)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(f'Comparison of Gradient and Newton Methods: {title} function', fontsize=16)
    plt.legend()
    plt.show()
