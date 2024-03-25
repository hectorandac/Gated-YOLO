import pandas as pd
import plotly.graph_objects as go

# Read the CSV file
df = pd.read_csv('/home/hectorandac/Documents/yolo-v6-size-invariant/YOLOv6/gate_proportions.csv')

# Number of columns (assuming 363 columns as mentioned)
num_columns = 109
num_rows = len(df)

# Sky blue color in RGB (scaled to 0-1)
sky_blue = (135/255, 206/255, 235/255)

# Create a 3D plot using Plotly
fig = go.Figure()

# Create data for 3D plot
for col_index in range(num_columns):
    xs = [col_index] * num_rows  # Column ID repeated for each row
    ys = list(range(num_rows))   # Row numbers
    zs = df.iloc[:, col_index]   # Percentages for each row in the column

    # Calculate the intensity and color for this row
    # Adjust the maximum intensity to 50%
    intensity = 0.25 + 0.50 * col_index / (num_columns - 1)
    col_color = 'rgba({},{},{},{})'.format(intensity * sky_blue[0] * 255,
                                           intensity * sky_blue[1] * 255,
                                           intensity * sky_blue[2] * 255,
                                           0.8)  # Adjust opacity as needed

    # Add the trace for each column
    fig.add_trace(go.Scatter3d(x=ys, y=xs, z=zs, mode='lines',
                               line=dict(color=col_color, width=8)))

# Update layout for a better view
fig.update_layout(scene=dict(
                    xaxis_title='Row Number',
                    yaxis_title='Column ID',
                    zaxis_title='Percentage'),
                  margin=dict(r=10, l=10, b=10, t=10))

# Display the plot
fig.show()
