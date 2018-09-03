from SALib.analyze import morris
from SALib.sample.morris import sample
import numpy as np
from multiprocessing import Pool, cpu_count
import hdmorrispp as hdmpp


def run_salib_morris(
    target_parameters,
    target_output_name,
    n_trajectories,
    num_levels,
    grid_jump,
    n_processors = cpu_count() - 1
    ):

    num_vars = len(target_parameters)
    parameter_names = list(target_parameters.keys())
    groups = None
    bounds = [target_parameters[name] for name in parameter_names]

    problem = {
        'num_vars': num_vars,
        'names': parameter_names,
        'groups': groups,
        'bounds': bounds
    }

    parameter_values = sample(
        problem,
        N=n_trajectories,
        num_levels=num_levels,
        grid_jump=grid_jump,
        optimal_trajectories=None
    )

    ys = None

    if n_processors > 1:
        chunks = np.array_split(parameter_values, n_processors)
        chunks = [(parameter_names, chunk, target_output_name) for chunk in chunks]
        pool = Pool(n_processors)
        results = pool.map(
            hdmpp.generate_outputs,
            chunks
            )
        ys = np.concatenate(results)
    else:
        ys = hdmpp.generate_outputs((parameter_names, parameter_values, target_output_name))

    sis = morris.analyze(
        problem,
        parameter_values,
        np.array(ys),
        conf_level=0.95,
        print_to_console=False,
        num_levels=num_levels,
        grid_jump=grid_jump,
        num_resamples=100
    )

    return problem, parameter_values, sis
