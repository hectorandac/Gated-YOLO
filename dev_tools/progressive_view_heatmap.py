import pandas as pd
import plotly.graph_objects as go

# Read the CSV file
df = pd.read_csv('/home/hectorandac/Documents/yolo-v6-size-invariant/YOLOv6/gate_proportions.csv')

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=list(range(df.shape[1])),  # Column IDs
        y=list(range(df.shape[0])),  # Row numbers
        colorscale='Blues'))

# Update the layout
fig.update_layout(
    title='Layer Activation Heatmap',
    xaxis_nticks=36,
    yaxis_nticks=36,
    xaxis_title='Column ID',
    yaxis_title='Row Number'
)

# Show the figure
fig.show()