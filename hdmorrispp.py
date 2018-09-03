import numpy as np
import hexamolldinch as hd


def generate_outputs(input):
   
    parameter_names, parameter_values, target_output_name = input

    parameters = hd.assign_parameters()
    schedule = hd.get_default_schedule()
    _, _, _, output = hd.pbpk(parameters, schedule)
    y_index = np.argmax(output[target_output_name])

    ys = []

    for inputs in parameter_values:
        
        for i in range(len(parameter_names)):
            setattr(parameters, parameter_names[i], inputs[i])

        _, _, _, output = hd.pbpk(parameters, schedule)

        ys.append(output[target_output_name][y_index])

    return ys
