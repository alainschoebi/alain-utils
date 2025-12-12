# NumPy
import numpy as np

# Python
from typing import Any, Optional, Literal
from pathlib import Path

# Plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    pass

# Utils
from alutils import Color
from alutils.decorators import requires_package
from alutils.loggers import get_logger
logger = get_logger(__name__)

PlotLayout = list[list[dict[str, Any]]]

@requires_package('plotly')
def build_plotly_plot(
        plot: PlotLayout,
        *,
        title: Optional[str] = "",
        height: Optional[int | None] = None,
        open_browser: Optional[bool] = True,
        output_html: Optional[str | Path | None] = None,
        hover_mode: Literal['x', 'y', 'x unified', 'y unified',
                            'closest', 'False'] = 'x unified',
        output_svg: Optional[str | Path] = None,
        bar_mode: Literal['group', 'overlay', 'stack', 'relative'] = 'group',
        bg_color: Color = Color("#E6E6E6"),
    ) -> go.Figure:
    """
    Builds a plotly plot from a 2D list of dictionaries. Each dictionary
    describes a subplot of the plot.

    Inputs
    ------
    plot: `list[list[dict[str, Any]]]`
        The 2D list of dictionaries describing the plot.

    Keys
    ----
    Each entry in the 2D list of dictionaries MUST have the following keys:

    title: `str`
        The title of the subplot.
    traces: `go.Scatter | go.Image ... | list[go.Scatter | go.Image | ...]`
        The trace(s) of the subplot.

    Each entry in the 2D list of dictionaries CAN have the optional keys:

    rowspan: `int`
        The number of rows spanned by the subplot.
    colspan: `int`
        the number of columns spanned by the subplot.
    xlim: `list[float, float]`
        the x-axis limits.
    ylim: `list[float, float]`
        the y-axis limits.
    secondary_ylim: `list[float, float]`
        the secondary y-axis limits.
    equal_aspect_ratio: `bool`
        whether to use equal aspect ratio for the subplot.
    viewpoint: `dict[str, float]`
        the viewpoint of the 3D plot.
    xlabel: `str`
        the label of the x-axis.
    ylabel: `str`
        the label of the y-axis.
    tick_format_x: `str`
        the tick format for the x-axis.
    tick_format_y: `str`
        the tick format for the y-axis.
    tick_format_secondary_y: `str`
        the tick format for the secondary y-axis.
    log_scale_x: `bool`
        whether to use a logarithmic scale for the x-axis.
    log_scale_y: `bool`
        whether to use a logarithmic scale for the y-axis.
    secondary_log_scale_y: `bool`
        whether to use a logarithmic scale for the secondary y-axis.
    secondary_ylabel: `str`
        the label of the secondary y-axis.
    secondary_y_axis_trace_idx: `list[int]`
        the indices of which traces should be plotted on the secondary y-axis.
    shared_x_axis_identifier: `str`
        the name of the shared x-axis. Note that the name only serves as
        identification and the actual string is irrelevant.
    shared_y_axis_identifier: `str`
        the name of the shared y-axis. Note that the name only serves as
        identification and the actual string is irrelevant.
    # TODO DO SHARED X-Y AXES joint

    Optional Inputs
    ---------------
    title: `str`
        The title of the plot.
    open_browser: `bool`
        Whether to open the plot in the browser. Default is `True`.
    output_html: `str | Path | None`
        The path to save the plot as an HTML file. Default is `None`.
    hover_mode: `x | y | x unified | y unified | closest | False`
        Hover mode for all the subplot. Default is `x unified`.
    bar_mode: TODO
        TODO
    """

    rows = len(plot)
    assert rows > 0 and "Plot cannot be empty"

    # Count number of columns (not very clean...)
    cols = []
    for i, row in enumerate(plot):
        col = len(row)
        for j in range(len(row)):
            if row[j] is not None and 'colspan' in row[j]:
                col += row[j]['colspan'] - 1
                for jj in range(j + 1, j + row[j]['colspan']):
                    if jj >= len(row):
                        break
                    if row[jj] is None:
                        col -= 1
                    else:
                        logger.error(
                            f"Invalid use of 'colspan'. The entry of the " +
                            f"plot at `plot[{i}][{jj}] should be `None` or " +
                            f"inexistent."
                        )
                        raise ValueError(
                            f"Invalid use of 'colspan'. The entry of the " +
                            f"plot at `plot[{i}][{jj}] should be `None` or " +
                            f"inexistent."
                        )
            if row[j] is not None and 'rowspan' in row[j]:
                rows = max(rows, i + row[j]['rowspan'])

        cols.append(col)

    cols = max(cols)

    # Boolean matrix indicating whether each subplot is "overwritten" by a
    # colspan/rowspan or not
    is_span = np.full((rows, cols), fill_value=False, dtype=bool)
    for i in range(len(plot)):
        for j in range(len(plot)):
            if j >= len(plot[i]) or plot[i][j] is None:
                continue
            rowspan, colspan = 1, 1
            if 'rowspan' in plot[i][j]:
                rowspan = plot[i][j]['rowspan']
            if 'colspan' in plot[i][j]:
                colspan = plot[i][j]['colspan']

            # Check validity
            for ii in range(i, i + rowspan):
                for jj in range(j, j + colspan):
                    if ii == i and jj == j:
                        continue

                    if ii >= len(plot) or \
                       jj >= len(plot[ii]) or \
                       plot[ii][jj] is None:
                       continue

                    logger.error(
                        f"Invalid use of 'rowspan' or 'colspan'. The entry " +
                        f"of the plot at `plot[{i}][{jj}] should be `None` " +
                        f"or inexistent."
                    )
                    raise ValueError(
                        f"Invalid use of 'rowspan' or 'colspan'. The entry " +
                        f"of the plot at `plot[{i}][{jj}] should be `None` " +
                        f"or inexistent."
                    )

            # Set span
            is_span[i:, j:][:rowspan, :colspan] = True
            is_span[i, j] = False


    # Build specifications and titles of the plot
    specs, titles = [], []
    shared_x_axis_identifiers = []
    shared_x_axis_identifiers_ref_subplot = []
    shared_y_axis_identifiers = []
    shared_y_axis_identifiers_ref_subplot = []
    scatter_3d_viewpoints, scatter_3d_counter = {}, 0
    for i in range(rows):
        specs_row = []
        for j in range(cols):

            # If subplot is "overwritten" by colspan/rowspan
            if is_span[i, j]:
                specs_row.append(None)
                continue

            # If subplot is not directly defined
            if i >= len(plot) or \
               j >= len(plot[i]):
                specs_row.append({})
                titles.append("?")
                continue

            if plot[i][j] is None:
                specs_row.append({})
                titles.append("")
                continue

            # Acces entry
            entry = plot[i][j]
            if not "title" in entry or not "traces" in entry:
                raise ValueError(f"The entry must have a 'title' and a " +
                                 f"'traces' key.")

            # Subplot title
            titles.append(entry["title"])

            # Check if any traces should be plotted
            if entry["traces"] is None or \
               isinstance(entry["traces"], list) and len(entry["traces"]) == 0:
               specs_row.append({})
               continue

            # Pick first trace
            trace = entry["traces"]
            if isinstance(entry["traces"], list):
                trace = entry["traces"][0]

            # Type of subplot
            if isinstance(trace, go.Contour):
                specs_row.append({"type": "contour"})
            elif isinstance(trace, (go.Scatter, go.Histogram)):
                specs_row.append({"type": "xy"})
            elif isinstance(trace, go.Image):
                specs_row.append({"type": "image"})
            elif isinstance(trace, go.Scatter3d) or \
                 isinstance(trace, go.Mesh3d):
                specs_row.append({"type": "scatter3d"})
                scatter_3d_counter += 1
                if 'viewpoint' in entry:
                    scene = f"scene{scatter_3d_counter}_camera"
                    scatter_3d_viewpoints[scene] = entry['viewpoint']
            elif isinstance(trace, go.Table):
                specs_row.append({"type": "table"})
            else:
                logger.error(f"Type '{type(trace)}' is not implemented.")
                raise NotImplementedError(f"Type '{type(trace)} is not " +
                                          f"implemented.")

            # Secondary y-axis
            if "secondary_y_axis_trace_idx" in entry:
                specs_row[-1].update({"secondary_y": True})

            # Rowspan and colspan
            rowspan, colspan = 0, 0
            if 'rowspan' in entry:
                rowspan = entry['rowspan']
                specs_row[-1]['rowspan'] = rowspan

            if 'colspan' in entry:
                colspan = entry['colspan']
                specs_row[-1]['colspan'] = colspan

            # Shared x-axes
            if 'shared_x_axis_identifier' in entry:
                identifier = entry['shared_x_axis_identifier']
                if not identifier in shared_x_axis_identifiers:
                    shared_x_axis_identifiers.append(identifier)
                    shared_x_axis_identifiers_ref_subplot.append((i, j))

            # Shared y-axes
            if 'shared_y_axis_identifier' in entry:
                identifier = entry['shared_y_axis_identifier']
                if not identifier in shared_y_axis_identifiers:
                    shared_y_axis_identifiers.append(identifier)
                    shared_y_axis_identifiers_ref_subplot.append((i, j))

        specs.append(specs_row)

    # Make plot
    fig = make_subplots(
        rows=rows, cols=cols, specs=specs, subplot_titles=titles
    )

    # Build plot
    axes_counter = 0
    for i, row in enumerate(plot):
        for j, entry in enumerate(row):

            if entry is None or entry["traces"] is None or \
               isinstance(entry["traces"], list) and len(entry["traces"]) == 0:
                if not is_span[i, j]: axes_counter += 1
                continue

            # Add subplots (multiple traces)
            if isinstance(entry["traces"], list):
                if len(entry["traces"]) == 0:
                    continue

                # Handle secondary y axes
                if "secondary_y_axis_trace_idx" in entry:
                    for idx, trace in enumerate(entry["traces"]):
                        secondary_y = idx in entry["secondary_y_axis_trace_idx"]
                        fig.add_trace(trace, row=i+1, col=j+1,
                                      secondary_y=secondary_y)
                else:
                     for trace in entry["traces"]:
                        fig.add_trace(trace, row=i+1, col=j+1)
            # Add subplots (single trace)
            else:
                fig.add_trace(entry["traces"], row=i+1, col=j+1)


            # Pick first trace
            trace = entry["traces"]
            if isinstance(entry["traces"], list):
                trace = entry["traces"][0]

            # Count number of usual x-y axes
            if not isinstance(trace, (go.Table, go.Mesh3d)):
                axes_counter += 1
                plot[i][j]['_axes_counter'] = axes_counter

            # Axes labels
            if 'xlabel' in entry:
                fig.update_xaxes(title_text=entry['xlabel'], row=i+1, col=j+1)
            if 'ylabel' in entry:
                fig.update_yaxes(title_text=entry['ylabel'], row=i+1, col=j+1)
            if 'secondary_ylabel' in entry:
                fig.update_yaxes(title_text=entry['secondary_ylabel'],
                                 row=i+1, col=j+1, secondary_y=True)

            # Axes scale
            if 'log_sale_x' in entry and entry['log_scale_x']:
                fig.update_xaxes(type='log', row=i+1, col=j+1)
            if 'log_scale_y' in entry and entry['log_scale_y']:
                fig.update_yaxes(type='log', row=i+1, col=j+1,
                                 secondary_y=False)
            if 'secondary_log_scale_y' in entry and \
               entry['secondary_log_scale_y']:
                fig.update_yaxes(type='log', row=i+1, col=j+1,
                                 secondary_y=True)

            # Shared x-axis
            if 'shared_x_axis_identifier' in entry:
                idx = shared_x_axis_identifiers \
                    .index(entry['shared_x_axis_identifier'])
                i_ref, j_ref = shared_x_axis_identifiers_ref_subplot[idx]
                if not (i, j) == (i_ref, j_ref):
                    fig.update_xaxes(
                        matches=f"x{plot[i_ref][j_ref]['_axes_counter']}",
                        row=i+1, col=j+1
                    )

            # Shared y-axis
            if 'shared_y_axis_identifier' in entry:
                idx = shared_y_axis_identifiers \
                    .index(entry['shared_y_axis_identifier'])
                i_ref, j_ref = shared_y_axis_identifiers_ref_subplot[idx]
                if not (i, j) == (i_ref, j_ref):
                    fig.update_yaxes(
                        matches=f"y{plot[i_ref][j_ref]['_axes_counter']}",
                        row=i+1, col=j+1
                    )

            # Axes limits
            if not isinstance(trace, (go.Scatter, go.Contour,
                                      go.Image, go.Histogram)) and \
                any([k in entry for k in ['xlim', 'ylim', 'secondary_ylim']]):
                raise ValueError(
                    f"Using properties `xlim`, `ylim` or `secondary_ylim` is "
                    f"only supported when using `Scatter`, `Contour`, `Image` "
                    f"or `Histogram` plots. Found plot type `{type(trace)}`."
                )

            if 'xlim' in entry:
                xlim = entry['xlim']
                if not isinstance(xlim, list) or not len(xlim) == 2 or \
                   None in xlim:
                    raise ValueError(f"The provided xlim='{xlim}' is " +
                                     f"invalid.")
                fig.update_xaxes(range=xlim, row=i+1, col=j+1)
            if 'ylim' in entry:
                ylim = entry['ylim']
                if not isinstance(ylim, list) or not len(ylim) == 2 or \
                    None in ylim:
                    raise ValueError(f"The provided ylim='{ylim}' is " +
                                     f"invalid.")
                fig.update_yaxes(range=ylim, row=i+1, col=j+1,
                                 secondary_y=False)
            if 'secondary_ylim' in entry:
                secondary_ylim = entry['secondary_ylim']
                if not isinstance(secondary_ylim, list) or \
                    not len(secondary_ylim) == 2 or None in secondary_ylim:
                    raise ValueError(f"The provided secondary_ylim='" +
                                     f"{secondary_ylim}' is invalid.")
                fig.update_yaxes(range=secondary_ylim,
                                 row=i+1, col=j+1, secondary_y=True)

            if 'tick_format_x' in entry:
                tick_format_x = entry['tick_format_x']
                fig.update_xaxes(
                    tickformat=tick_format_x, row=i+1, col=j+1
                )

            if 'tick_format_y' in entry:
                tick_format_y = entry['tick_format_y']
                fig.update_yaxes(
                    tickformat=tick_format_y, row=i+1, col=j+1
                )

            if 'tick_format_secondary_y' in entry:
                tick_format_secondary_y = entry['tick_format_secondary_y']
                fig.update_yaxes(
                    tickformat=tick_format_secondary_y,
                    row=i+1, col=j+1, secondary_y=True
                )

            # Aspect ratio 1:1 for traces with go.Scatter as first plot
            if 'equal_aspect_ratio' in entry and ( \
                    not entry['equal_aspect_ratio'] or \
                    not isinstance(trace, (go.Scatter))
                ):
                raise ValueError(
                    "If `equal_aspect_ratio` is provided it must be set to " +
                    "`True` and the traces must start with a `go.Scatter` plot."
                )

            if 'equal_aspect_ratio' in entry and \
               isinstance(trace, (go.Scatter)):
                fig.update_yaxes(scaleanchor=f'x{axes_counter}',
                                 row=i+1, col=j+1)

            # Automatic axes bounds for go.Image as first trace
            if isinstance(trace, go.Image):
                H, W = trace.z.shape[:2]
                if not 'xlim' in entry and not 'ylim' in entry and \
                    not 'secondary_ylim' in entry:
                    fig.update_xaxes(range=[0, W], row=i+1, col=j+1)
                    fig.update_yaxes(range=[H, 0], row=i+1, col=j+1)

            # Automatic aspect ratio 1:1 for go.Contour as first trace
            elif isinstance(trace, go.Contour):
                fig.update_xaxes(constrain='domain', row=i+1, col=j+1)
                fig.update_yaxes(constrain='domain',
                                 scaleanchor=f'x{axes_counter}',
                                 row=i+1, col=j+1)

    # Layout
    if height is None:
        height = 400 * rows
    fig.update_layout(
        title=title,
        showlegend=False,
        height=height,
        **scatter_3d_viewpoints,
        hovermode=hover_mode,
        barmode=bar_mode,
        plot_bgcolor=bg_color.rgb_string(),
    )

    # Save HTML file
    if output_html is not None:
        if not Path(output_html).parent.exists():
            logger.error(
                f"The directory '{Path(output_html).parent}' does not exist. " +
                f"Cannot save Plotly plot."
            )
            raise FileNotFoundError(
                f"The directory '{Path(output_html).parent}' does not exist. " +
                f"Cannot save Plotly plot."
            )
        fig.write_html(str(output_html), include_mathjax='cdn')

    # Save SVG file
    if output_svg is not None:
        try:
            import kaleido
        except ImportError as e:
            raise ImportError(
                "The package `kaleido` is required to export Plotly plots as "
                "SVG files. Please install it using `pip install kaleido`."
            )

        if not Path(output_svg).parent.exists():
            logger.error(
                f"The directory '{Path(output_svg).parent}' does not exist. " +
                f"Cannot save Plotly plot."
            )
            raise FileNotFoundError(
                f"The directory '{Path(output_svg).parent}' does not exist. " +
                f"Cannot save Plotly plot."
            )
        fig.write_image(str(output_svg), format='svg')

    # Open browser
    if open_browser:
        fig.show()

    # Log
    logger.info(f"Successfully built Plotly plot `{title}` to `{output_html}`.")
    logger.debug(fig.layout) # log layout for debugging purposes

    return fig
