# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from datetime import datetime
# from io import BytesIO
# import base64
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.express as px
# from io import BytesIO
# import base64
#
#
#
#
# def create_calendar_df(df: pd.DataFrame, start_date: str, end_date: str):
#     """Returns a Series indexed by date, covering the date range, with zeros for missing days."""
#     df = df.copy()
#     df['Date'] = pd.to_datetime(df['Date'])
#     df = df.set_index('Date').resample('D')['Total Pallets'].sum().to_frame()
#     full_range = pd.date_range(start=start_date, end=end_date)
#     df = df.reindex(full_range, fill_value=0)
#     df.index.name = 'Date'
#     return df['Total Pallets']
#
#
# def plot_single_calendar_with_months(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
#                                      colorscale='YlGn', title="Calendar Heatmap"):
#     """Plots a single calendar heatmap with visible month boundaries and month labels using Plotly."""
#     num_days = (end - start).days + 1
#     start_weekday = start.weekday()
#     num_weeks = ((num_days + start_weekday - 1) // 7) + 2
#     data = np.full((7, num_weeks), np.nan)
#
#     month_ticks = {}
#     month_positions = []
#
#     for date in pd.date_range(start, end):
#         week = ((date - start).days + start_weekday) // 7
#         weekday = date.weekday()
#         value = series.get(date, np.nan)
#         data[weekday, week] = value
#         if date.day == 1:
#             month_name = date.strftime('%b %Y')
#             month_ticks[week] = month_name
#             month_positions.append(week)
#
#     # Create heatmap
#     fig = go.Figure(data=go.Heatmap(
#         z=data,
#         colorscale=colorscale,
#         showscale=True,
#         hoverongaps=False,
#         colorbar=dict(
#             title="Total Pallets Shipped",
#             titleside="bottom",
#             orientation="h",
#             y=-0.1,
#             len=0.6,
#             x=0.2
#         )
#     ))
#
#     # Configure layout
#     fig.update_layout(
#         title=title,
#         xaxis=dict(
#             tickmode='array',
#             tickvals=list(month_ticks.keys()),
#             ticktext=list(month_ticks.values()),
#             tickangle=45,
#             side='bottom',
#             showgrid=False,
#             zeroline=False
#         ),
#         yaxis=dict(
#             tickmode='array',
#             tickvals=list(range(7)),
#             ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
#             showgrid=False,
#             zeroline=False,
#             autorange='reversed'
#         ),
#         width=max(800, num_weeks * 30),
#         height=300,
#         margin=dict(t=80, b=100, l=80, r=50)
#     )
#
#     # Add month boundary lines
#     shapes = []
#     for tick in month_ticks:
#         shapes.append(dict(
#             type="rect",
#             x0=tick - 0.5,
#             y0=-0.5,
#             x1=tick + 0.5,
#             y1=6.5,
#             line=dict(color="black", width=1),
#             fillcolor="rgba(0,0,0,0)"
#         ))
#
#     fig.update_layout(shapes=shapes)
#
#     return fig
#
#
# def plot_dual_calendar_heatmaps(df1: pd.DataFrame, df2: pd.DataFrame, save_path="calendar_dual_heatmap.png"):
#     """Create dual calendar heatmaps using Plotly subplots."""
#     # Determine earliest and latest year in combined data
#     min_year = min(df1['Date'].min().year, df2['Date'].min().year)
#     max_year = max(df1['Date'].max().year, df2['Date'].max().year)
#
#     # Pad to full years
#     start_date = pd.to_datetime(f"{min_year}-01-01")
#     end_date = pd.to_datetime(f"{max_year}-12-31")
#
#     # Normalize input data to full-year span
#     series1 = create_calendar_df(df1, start_date, end_date)
#     series2 = create_calendar_df(df2, start_date, end_date)
#
#     # Create data matrices
#     num_days = (end_date - start_date).days + 1
#     start_weekday = start_date.weekday()
#     num_weeks = ((num_days + start_weekday - 1) // 7) + 2
#
#     data1 = np.full((7, num_weeks), np.nan)
#     data2 = np.full((7, num_weeks), np.nan)
#     month_ticks = {}
#
#     for date in pd.date_range(start_date, end_date):
#         week = ((date - start_date).days + start_weekday) // 7
#         weekday = date.weekday()
#
#         value1 = series1.get(date, np.nan)
#         value2 = series2.get(date, np.nan)
#
#         data1[weekday, week] = value1
#         data2[weekday, week] = value2
#
#         if date.day == 1:
#             month_name = date.strftime('%b %Y')
#             month_ticks[week] = month_name
#     print()
#     # Create subplot figure
#     fig = make_subplots(
#         rows=2, cols=1,
#         subplot_titles=("Calendar Heatmap: Original Shipments", "Calendar Heatmap: Consolidated Shipments"),
#         vertical_spacing=0.15,
#         shared_xaxes=True
#     )
#
#     # Add heatmaps
#     fig.add_trace(
#         go.Heatmap(
#             z=data1,
#             colorscale='YlGn',
#             showscale=False,
#             hoverongaps=False,
#             name="Original"
#         ),
#         row=1, col=1
#     )
#
#     fig.add_trace(
#         go.Heatmap(
#             z=data2,
#             colorscale='YlGn',
#             showscale=True,
#             hoverongaps=False,
#             name="Consolidated",
#             colorbar=dict(
#                 title="Total Pallets Shipped",
#                 titleside="bottom",
#                 orientation="h",
#                 y=-0.15,
#                 len=0.6,
#                 x=0.2
#             )
#         ),
#         row=2, col=1
#     )
#
#     # Configure layout
#     fig.update_layout(
#         width=max(1200, num_weeks * 30),
#         height=600,
#         margin=dict(t=100, b=150, l=80, r=50)
#     )
#
#     # Configure axes
#     weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
#
#     # Update y-axes
#     fig.update_yaxes(
#         tickmode='array',
#         tickvals=list(range(7)),
#         ticktext=weekday_labels,
#         showgrid=False,
#         zeroline=False,
#         autorange='reversed',
#         row=1, col=1
#     )
#
#     fig.update_yaxes(
#         tickmode='array',
#         tickvals=list(range(7)),
#         ticktext=weekday_labels,
#         showgrid=False,
#         zeroline=False,
#         autorange='reversed',
#         row=2, col=1
#     )
#
#     # Update x-axes
#     fig.update_xaxes(
#         tickmode='array',
#         tickvals=list(month_ticks.keys()),
#         ticktext=list(month_ticks.values()),
#         tickangle=45,
#         showgrid=False,
#         zeroline=False,
#         row=2, col=1
#     )
#
#     # Add month boundary lines
#     shapes = []
#     for tick in month_ticks:
#         # Lines for first subplot
#         shapes.append(dict(
#             type="rect",
#             x0=tick - 0.5,
#             y0=-0.5,
#             x1=tick + 0.5,
#             y1=6.5,
#             line=dict(color="black", width=1),
#             fillcolor="rgba(0,0,0,0)",
#             xref="x",
#             yref="y"
#         ))
#         # Lines for second subplot
#         shapes.append(dict(
#             type="rect",
#             x0=tick - 0.5,
#             y0=-0.5,
#             x1=tick + 0.5,
#             y1=6.5,
#             line=dict(color="black", width=1),
#             fillcolor="rgba(0,0,0,0)",
#             xref="x2",
#             yref="y2"
#         ))
#
#     fig.update_layout(shapes=shapes)
#
#     # Save to file
#     # fig.write_html(save_path)
#
#     # Return as base64 and figure
#     # image_bytes = BytesIO()
#     # fig.write_image(image_bytes, format='png', width=1200, height=600)
#     # image_bytes.seek(0)
#
#     # return image_bytes, fig
#     return fig
#
#
# def plot_dual_calendar_heatmaps_filtered(df1: pd.DataFrame, df2: pd.DataFrame, start_date, end_date,
#                                          save_path="calendar_dual_heatmap_filtered.html"):
#     """Create filtered dual calendar heatmaps using Plotly subplots."""
#     start_date = pd.to_datetime(start_date)
#     end_date = pd.to_datetime(end_date)
#
#     # Normalize input data to specified date range
#     series1 = create_calendar_df(df1, start_date, end_date)
#     series2 = create_calendar_df(df2, start_date, end_date)
#
#     # Create data matrices
#     num_days = (end_date - start_date).days + 1
#     start_weekday = start_date.weekday()
#     num_weeks = ((num_days + start_weekday - 1) // 7) + 2
#
#     data1 = np.full((7, num_weeks), np.nan)
#     data2 = np.full((7, num_weeks), np.nan)
#     month_ticks = {}
#
#     for date in pd.date_range(start_date, end_date):
#         week = ((date - start_date).days + start_weekday) // 7
#         weekday = date.weekday()
#
#         value1 = series1.get(date, np.nan)
#         value2 = series2.get(date, np.nan)
#
#         data1[weekday, week] = value1
#         data2[weekday, week] = value2
#
#         if date.day == 1:
#             month_name = date.strftime('%b %Y')
#             month_ticks[week] = month_name
#
#     # Create subplot figure
#     fig = make_subplots(
#         rows=2, cols=1,
#         subplot_titles=("Calendar Heatmap: Original Shipments", "Calendar Heatmap: Consolidated Shipments"),
#         vertical_spacing=0.15,
#         shared_xaxes=True
#     )
#
#     # Add heatmaps
#     fig.add_trace(
#         go.Heatmap(
#             z=data1,
#             colorscale='YlGn',
#             showscale=False,
#             hoverongaps=False,
#             name="Original"
#         ),
#         row=1, col=1
#     )
#
#     fig.add_trace(
#         go.Heatmap(
#             z=data2,
#             colorscale='YlGn',
#             showscale=True,
#             hoverongaps=False,
#             name="Consolidated",
#             colorbar=dict(
#                 title="Total Pallets Shipped",
#                 titleside="bottom",
#                 orientation="h",
#                 y=-0.15,
#                 len=0.6,
#                 x=0.2
#             )
#         ),
#         row=2, col=1
#     )
#
#     # Configure layout
#     fig.update_layout(
#         width=max(1200, num_weeks * 30),
#         height=600,
#         margin=dict(t=100, b=150, l=80, r=50)
#     )
#
#     # Configure axes
#     weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
#
#     # Update y-axes
#     fig.update_yaxes(
#         tickmode='array',
#         tickvals=list(range(7)),
#         ticktext=weekday_labels,
#         showgrid=False,
#         zeroline=False,
#         autorange='reversed',
#         row=1, col=1
#     )
#
#     fig.update_yaxes(
#         tickmode='array',
#         tickvals=list(range(7)),
#         ticktext=weekday_labels,
#         showgrid=False,
#         zeroline=False,
#         autorange='reversed',
#         row=2, col=1
#     )
#
#     # Update x-axes
#     fig.update_xaxes(
#         tickmode='array',
#         tickvals=list(month_ticks.keys()),
#         ticktext=list(month_ticks.values()),
#         tickangle=45,
#         showgrid=False,
#         zeroline=False,
#         row=2, col=1
#     )
#
#     # Add month boundary lines
#     shapes = []
#     for tick in month_ticks:
#         # Lines for first subplot
#         shapes.append(dict(
#             type="rect",
#             x0=tick - 0.5,
#             y0=-0.5,
#             x1=tick + 0.5,
#             y1=6.5,
#             line=dict(color="black", width=1),
#             fillcolor="rgba(0,0,0,0)",
#             xref="x",
#             yref="y"
#         ))
#         # Lines for second subplot
#         shapes.append(dict(
#             type="rect",
#             x0=tick - 0.5,
#             y0=-0.5,
#             x1=tick + 0.5,
#             y1=6.5,
#             line=dict(color="black", width=1),
#             fillcolor="rgba(0,0,0,0)",
#             xref="x2",
#             yref="y2"
#         ))
#
#     fig.update_layout(shapes=shapes)
#
#     # Save to file
#     fig.write_html(save_path)
#
#     # Return as base64
#     image_bytes = BytesIO()
#     fig.write_image(image_bytes, format='png', width=1200, height=600)
#     image_bytes.seek(0)
#
#     return image_bytes





# def plot_single_calendar_with_months(ax, series: pd.Series, start: pd.Timestamp, end: pd.Timestamp, cmap='YlGn'):
#     """Plots a single calendar heatmap with visible month boundaries and month labels."""
#     num_days = (end - start).days + 1
#     start_weekday = start.weekday()
#     num_weeks = ((num_days + start_weekday - 1) // 7) + 2
#     data = np.full((7, num_weeks), np.nan)
#
#     month_ticks = {}
#     for date in pd.date_range(start, end):
#         week = ((date - start).days + start_weekday) // 7
#         weekday = date.weekday()
#         value = series.get(date, np.nan)
#         data[weekday, week] = value
#         if date.day == 1:
#             month_name = date.strftime('%b %Y')
#             month_ticks[week] = month_name
#
#     im = ax.imshow(data, aspect='auto', cmap=cmap, origin='upper')
#     ax.set_yticks(range(7))
#     ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
#     ax.set_xticks(list(month_ticks.keys()))
#     ax.set_xticklabels(list(month_ticks.values()), rotation=45, ha='right')
#     ax.tick_params(axis='x', which='both', bottom=False, top=False)
#
#     for tick in month_ticks:
#         ax.add_patch(patches.Rectangle((tick - 0.5, -0.5), 1, 7, fill=False, edgecolor='black', linewidth=1))
#
#     return im
# def plot_dual_calendar_heatmaps(df1: pd.DataFrame, df2: pd.DataFrame, save_path="calendar_dual_heatmap.png"):
#     # Determine earliest and latest year in combined data
#     min_year = min(df1['Date'].min().year, df2['Date'].min().year)
#     max_year = max(df1['Date'].max().year, df2['Date'].max().year)
#
#     # Pad to full years
#     start_date = pd.to_datetime(f"{min_year}-01-01")
#     end_date = pd.to_datetime(f"{max_year}-12-31")
#
#     # Normalize input data to full-year span
#     series1 = create_calendar_df(df1, start_date, end_date)
#     series2 = create_calendar_df(df2, start_date, end_date)
#
#     # Plotting
#     fig, axes = plt.subplots(2, 1, figsize=(max(16, (end_date - start_date).days // 20), 6), constrained_layout=True)
#     im1 = plot_single_calendar_with_months(axes[0], series1, start_date, end_date)
#     im2 = plot_single_calendar_with_months(axes[1], series2, start_date, end_date)
#
#     axes[0].set_title("Calendar Heatmap: Original Shipments")
#     axes[1].set_title("Calendar Heatmap: Consolidated Shipments")
#
#     # Shared colorbar with legend title
#     cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.6, pad=0.05)
#     cbar.set_label('Total Pallets Shipped', fontsize=12, fontweight='bold', family='sans-serif', labelpad=10)
#
#
#     # Save to file
#     plt.savefig(save_path, dpi=300)
#
#     # Return as base64
#     image_bytes = BytesIO()
#     plt.savefig(image_bytes, format='png')
#     image_bytes.seek(0)
#     plt.close(fig)
#
#     return image_bytes, fig

# def plot_dual_calendar_heatmaps_filtered(df1: pd.DataFrame, df2: pd.DataFrame,start_date,end_date, save_path="calendar_dual_heatmap.png"):
#     # Determine earliest and latest year in combined data
#     min_year = min(df1['Date'].min().year, df2['Date'].min().year)
#     max_year = max(df1['Date'].max().year, df2['Date'].max().year)
#
#     # Pad to full years
#     # start_date = pd.to_datetime(f"{min_year}-01-01")
#     # end_date = pd.to_datetime(f"{max_year}-12-31")
#
#     start_date = pd.to_datetime(start_date)
#     end_date = pd.to_datetime(end_date)
#
#     # Normalize input data to full-year span
#     series1 = create_calendar_df(df1, start_date, end_date)
#     series2 = create_calendar_df(df2, start_date, end_date)
#
#     # Plotting
#     fig, axes = plt.subplots(2, 1, figsize=(max(16, (end_date - start_date).days // 20), 6), constrained_layout=True)
#     im1 = plot_single_calendar_with_months(axes[0], series1, start_date, end_date)
#     im2 = plot_single_calendar_with_months(axes[1], series2, start_date, end_date)
#
#     axes[0].set_title("Calendar Heatmap: Original Shipments")
#     axes[1].set_title("Calendar Heatmap: Consolidated Shipments")
#
#     # Shared colorbar with legend title
#     cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.6, pad=0.05)
#     cbar.set_label('Total Pallets Shipped', fontsize=12, fontweight='bold', family='sans-serif', labelpad=10)
#
#
#     # Save to file
#     plt.savefig(save_path, dpi=300)
#
#     # Return as base64
#     image_bytes = BytesIO()
#     plt.savefig(image_bytes, format='png')
#     image_bytes.seek(0)
#     plt.close(fig)
#
#     return image_bytes


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime


def create_calendar_df(df: pd.DataFrame, start_date: str, end_date: str):
    """Returns a Series indexed by date, covering the date range, with zeros for missing days."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').resample('D')['Total Pallets'].sum().to_frame()
    full_range = pd.date_range(start=start_date, end=end_date)
    df = df.reindex(full_range, fill_value=0)
    df.index.name = 'Date'
    return df['Total Pallets']


def create_calendar_matrix(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp):
    """Creates a matrix for calendar heatmap with proper positioning."""
    num_days = (end - start).days + 1
    start_weekday = start.weekday()
    num_weeks = ((num_days + start_weekday - 1) // 7) + 2
    data = np.full((7, num_weeks), np.nan)

    # Store date information for hover text
    date_matrix = np.full((7, num_weeks), None, dtype=object)

    month_positions = {}
    for date in pd.date_range(start, end):
        week = ((date - start).days + start_weekday) // 7
        weekday = date.weekday()
        value = series.get(date, np.nan)
        data[weekday, week] = value
        date_matrix[weekday, week] = date.strftime('%Y-%m-%d')

        if date.day == 1:
            month_name = date.strftime('%b %Y')
            month_positions[week] = month_name

    return data, date_matrix, month_positions


def create_single_calendar_heatmap(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp, title: str):
    """Creates a single calendar heatmap using plotly."""
    data, date_matrix, month_positions = create_calendar_matrix(series, start, end)

    # Create hover text
    hover_text = []
    for i in range(data.shape[0]):
        hover_row = []
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]) and date_matrix[i, j] is not None:
                hover_row.append(f'Date: {date_matrix[i, j]}<br>Total Pallets: {data[i, j]:.0f}')
            else:
                hover_row.append('')
        hover_text.append(hover_row)

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale='YlGn',
        hovertemplate='%{text}<extra></extra>',
        text=hover_text,
        showscale=True,
        colorbar=dict(title="Total Pallets Shipped")
    ))

    # Set layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="",
            tickmode='array',
            tickvals=list(month_positions.keys()),
            ticktext=list(month_positions.values()),
            tickangle=45,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="",
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            showgrid=False,
            zeroline=False
        ),
        width=max(800, len(month_positions) * 60),
        height=300
    )

    # Add month boundary lines
    for tick in month_positions.keys():
        fig.add_vline(x=tick - 0.5, line=dict(color="black", width=1))

    return fig


def plot_dual_calendar_heatmaps(df1: pd.DataFrame, df2: pd.DataFrame, save_path="calendar_dual_heatmap.html"):
    """Creates dual calendar heatmaps using plotly with subplots."""
    # Determine earliest and latest year in combined data
    min_year = min(df1['Date'].min().year, df2['Date'].min().year)
    max_year = max(df1['Date'].max().year, df2['Date'].max().year)

    # Pad to full years
    start_date = pd.to_datetime(f"{min_year}-01-01")
    end_date = pd.to_datetime(f"{max_year}-12-31")

    # Normalize input data to full-year span
    series1 = create_calendar_df(df1, start_date, end_date)
    series2 = create_calendar_df(df2, start_date, end_date)

    # Create matrices for both calendars
    data1, date_matrix1, month_positions = create_calendar_matrix(series1, start_date, end_date)
    data2, date_matrix2, _ = create_calendar_matrix(series2, start_date, end_date)

    # Create hover text for both calendars
    hover_text1 = []
    hover_text2 = []

    for i in range(data1.shape[0]):
        hover_row1 = []
        hover_row2 = []
        for j in range(data1.shape[1]):
            if not np.isnan(data1[i, j]) and date_matrix1[i, j] is not None:
                hover_row1.append(f'Date: {date_matrix1[i, j]}<br>Total Pallets: {data1[i, j]:.0f}')
            else:
                hover_row1.append('')

            if not np.isnan(data2[i, j]) and date_matrix2[i, j] is not None:
                hover_row2.append(f'Date: {date_matrix2[i, j]}<br>Total Pallets: {data2[i, j]:.0f}')
            else:
                hover_row2.append('')
        hover_text1.append(hover_row1)
        hover_text2.append(hover_row2)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Calendar Heatmap: Original Shipments", "Calendar Heatmap: Consolidated Shipments"),
        vertical_spacing=0.3
    )

    # Add first heatmap
    fig.add_trace(
        go.Heatmap(
            z=data1,
            colorscale='YlGn',
            hovertemplate='%{text}<extra></extra>',
            text=hover_text1,
            showscale=False,
            name="Original"
        ),
        row=1, col=1
    )

    # Add second heatmap
    fig.add_trace(
        go.Heatmap(
            z=data2,
            colorscale='YlGn',
            hovertemplate='%{text}<extra></extra>',
            text=hover_text2,
            showscale=True,
            colorbar=dict(title="Total Pallets Shipped", y=0.5, len=0.9),
            name="Consolidated"
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        width=max(5000, len(month_positions) * 60),
        height=600,
        title_text="",
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='white',
        shapes=[
            # Border for first subplot (row=1, col=1)
            dict(
                type="rect",
                xref="x domain", yref="y domain",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="black", width=1),
                layer="above"
            ),
            # Border for second subplot (row=2, col=1)
            dict(
                type="rect",
                xref="x2 domain", yref="y2 domain",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="black", width=1),
                layer="above"
            )
        ]
    )

    # Update x-axes
    for i in [1, 2]:
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(month_positions.keys()),
            ticktext=list(month_positions.values()),
            tickangle=-45,
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=14, family='Arial', color='black'),
            row=i, col=1
        )

    # Update y-axes
    for i in [1, 2]:
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=14, family='Arial', color='black'),
            row=i, col=1
        )

    # Add month boundary lines for both subplots
    for tick in month_positions.keys():
        fig.add_vline(x=tick - 2, line=dict(color="black", width=1), row=1, col=1)
        fig.add_vline(x=tick - 2, line=dict(color="black", width=1), row=2, col=1)

    return fig



