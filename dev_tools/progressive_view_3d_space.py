import pandas as pd
import plotly.graph_objects as go

# Read the CSV file
df = pd.read_csv('/home/hectorandac/Documents/yolo-v6-size-invariant/YOLOv6/gate_proportions.csv')

# Create the 3D surface plot
fig = go.Figure(data=[go.Surface(z=df.values, x=list(range(df.shape[1])), y=list(range(df.shape[0])))])
fig.update_layout(
    title='3D Surface Plot of Layer Activations',
    autosize=True,
    margin=dict(l=0, r=0, b=0, t=30)  # Reduce margins to a minimum
)

# Update axes titles
fig.update_layout(scene=dict(
    xaxis_title='Column ID',
    yaxis_title='Row Number',
    zaxis_title='Activation Level'
))

# Show the figure
fig.show()
