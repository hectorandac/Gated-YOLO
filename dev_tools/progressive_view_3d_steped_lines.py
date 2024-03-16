import pandas as pd
import plotly.graph_objects as go

# Read the CSV file
df = pd.read_csv('./gate_proportions.csv')

# Number of columns (assuming 363 columns as mentioned)
num_columns = 151
num_rows = len(df)

# Sky blue color in RGB (scaled to 0-1)
sky_blue = (135/255, 206/255, 235/255)

# Create a 3D plot using Plotly
fig = go.Figure()

# Create data for 3D plot
for row_index in range(num_rows):
    xs = list(range(num_columns))   # Column IDs
    ys = [row_index] * num_columns  # Row Number repeated for each column
    zs = df.iloc[row_index, :]      # Percentages for each column in the row

    # Calculate the intensity and color for this row
    intensity = 0.25 + 0.25 * row_index / (num_rows)
    row_color = 'rgba({},{},{},{})'.format(intensity * sky_blue[0] * 255,
                                           intensity * sky_blue[1] * 255,
                                           intensity * sky_blue[2] * 255,
                                           0.8)  # Adjust opacity as needed

    # Add the trace for each row
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                               line=dict(color=row_color, width=8)))

# Update layout for a better view
fig.update_layout(scene=dict(
                    xaxis_title='Column ID',
                    yaxis_title='Row Number',
                    zaxis_title='Percentage'),
                  margin=dict(r=10, l=10, b=10, t=10))

# Display the plot
fig.show()
