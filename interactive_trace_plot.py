import plotly.express as px
import pandas as pd


df = pd.read_csv("C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/hyperparameter_tuning/uncongested/botsort_by_claude_uc_tracking_data.csv")
df = df.sort_values("frame").reset_index(drop=True)


# Plot lines for each track_id
fig = px.line(
    df,
    x='center_x',
    y='center_y',
    color='track_id',       # Each track gets a distinct color
    line_group='track_id',  # Ensures points for the same track are connected
    hover_data=['track_id', 'frame'],
    markers=True,           # Show markers at each point (optional)
)

# Fix axis ranges
fig.update_xaxes(range=[0, 9760])
fig.update_yaxes(range=[-2000, 2000])

fig.update_layout(title='Interactive Trace Lines')
fig.show()
