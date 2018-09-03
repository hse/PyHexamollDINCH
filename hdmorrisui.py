import os
import sys
import warnings
import ipywidgets as widgets
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
import matplotlib.pyplot as plt
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, sample_histograms
import hexamolldinch as hd
import hdmorris as hdm

warnings.simplefilter('ignore')

no_selection = "[None]"


def createControlRow(row_index, parameter_options):
    parameter_dropdown = widgets.Dropdown(
        options=parameter_options, value=no_selection)
    parameter_dropdown.rowIndex = row_index

    ft_LB = widgets.FloatText(disabled=True)
    ft_UB = widgets.FloatText(disabled=True)

    return [parameter_dropdown, ft_LB, ft_UB]


def canRunMorris(control_rows):
    names = [
        control_row[0].value
        for control_row in control_rows
        if control_row[0].value != no_selection
    ]

    if len(names) != len(set(names)):
        return False

    bounds = [
        (control_row[1].value, control_row[2].value)
        for control_row in control_rows
        if control_row[0].value != no_selection
    ]
    bounds = [bound for bound in bounds if bound[0] < bound[1]]
    return len(bounds) > 1


def onChangeSelectedParameter(change, parameters, control_rows, run_morris_button):
    selected_parameter_name = change.new
    value = 0.
    disabled = True
    if hasattr(parameters, selected_parameter_name):
        value = getattr(parameters, selected_parameter_name)
        disabled = False
    row_index = getattr(change.owner, 'rowIndex')
    control_row = control_rows[row_index]
    control_row[1].value = value * 0.95
    control_row[1].disabled = disabled
    control_row[2].value = value * 1.05
    control_row[2].disabled = disabled
    run_morris_button.disabled = not canRunMorris(control_rows)


def runMorris(
    control_rows,
    output_dropdown, 
    trajectories_text, 
    levels_text,
    grid_jump_text,
    run_morris_button,
    results_output
    ):
    inputs = [
        (control_row[0].value, control_row[1].value, control_row[2].value)
        for control_row in control_rows
        if control_row[0].value != no_selection
    ]

    target_parameters = dict([
        (input[0], (input[1], input[2]))
        for input in inputs
    ])

    target_output_name = output_dropdown.value

    n_trajectories = trajectories_text.value
    num_levels = levels_text.value
    grid_jump = grid_jump_text.value

    run_morris_button.disabled = True

    try:
        problem, parameter_values, sis = hdm.run_salib_morris(
            target_parameters,
            target_output_name,
            n_trajectories,
            num_levels,
            grid_jump,
            os.cpu_count()
        )
    except:
        with results_output:
            print("Invalid input", file=sys.stderr)
        return
    finally:
        run_morris_button.disabled = False

    from base64 import b64encode
    header = ','.join([name for name in target_parameters.keys()])
    lines = [','.join([str(d) for d in inputs]) for inputs in parameter_values]
    csv = header + os.linesep + os.linesep.join(lines)
    payload = b64encode(csv.encode())
    payload = payload.decode()
    fileName = "parameter_values.csv"
    title = "Download design"
    html = f'<a download="{fileName}" href="data:text/csv;charset=utf-8;base64,{payload}" target="_blank">{title}</a>'
    link = widgets.HTML(html)

    results_output.clear_output()

    with results_output:
        print("{0:<30} {1:>10} {2:>10} {3:>15} {4:>10}".format(
            "Parameter",
            "Mu_Star",
            "Mu",
            "Mu_Star_Conf",
            "Sigma")
        )
        num_vars = problem['num_vars']
        for j in list(range(num_vars)):
            print("{0:30} {1:10.3f} {2:10.3f} {3:15.3f} {4:10.3f}".format(
                sis['names'][j],
                sis['mu_star'][j],
                sis['mu'][j],
                sis['mu_star_conf'][j],
                sis['sigma'][j])
            )

        display(link)

        fig1, (ax1, ax2) = plt.subplots(1, 2)
        horizontal_bar_plot(ax1, sis, {}, sortby='mu_star')
        covariance_plot(ax2, sis, {})

        fig2 = plt.figure()
        sample_histograms(fig2, parameter_values, problem, {'color': 'y'})

        fig1.tight_layout()

        show_inline_matplotlib_plots()


def createMorrisUI():
    parameters = hd.assign_parameters()
    schedule = hd.get_default_schedule()
    _, _, _, output = hd.pbpk(parameters, schedule)
    output_options = list(output.keys())
    output_options.sort()

    parameter_names = [
        p for p in dir(parameters)
        if not p.startswith("_") and not p.startswith("t_")
    ]

    parameter_options = [no_selection] + parameter_names

    control_rows = [createControlRow(i, parameter_options) for i in range(5)]

    parameter_column = widgets.VBox(
        [widgets.Label(value="Parameter")] +
        [control_row[0] for control_row in control_rows]
    )
    lower_bound_column = widgets.VBox(
        [widgets.Label(value="Lower Bound")] +
        [control_row[1] for control_row in control_rows]
    )
    upper_bound_column = widgets.VBox(
        [widgets.Label(value="Upper Bound")] +
        [control_row[2] for control_row in control_rows]
    )
    parameters_ui = widgets.HBox(
        [parameter_column, lower_bound_column, upper_bound_column])

    output_label = widgets.Label(value="Target output:")
    output_dropdown = widgets.Dropdown(options=output_options)

    run_morris_button = widgets.Button(
        description='Run Morris',
        layout={'width': '150px'},
        disabled=True
    )

    for control_row in control_rows:
        control_row[0].observe(lambda c: onChangeSelectedParameter(
            c, parameters, control_rows, run_morris_button), 'value')

    trajectories_text = widgets.IntText(
        value=10,
        description="Trajectories:"
    )

    levels_text = widgets.IntText(
        value=4,
        description="Levels:"
    )

    grid_jump_text = widgets.IntText(
        value=2,
        description="Grid Jump:"
    )

    results_output = widgets.Output(layout={'border': '1px solid black'})

    run_morris_button.on_click(
        lambda b: 
            runMorris(
                control_rows,
                output_dropdown,
                trajectories_text,
                levels_text,
                grid_jump_text,
                run_morris_button,
                results_output
            )
    )

    ui = (
        parameters_ui,
        widgets.HBox([output_label, output_dropdown]),
        widgets.HBox([trajectories_text, levels_text, grid_jump_text]),
        run_morris_button,
        results_output
    )

    return ui
