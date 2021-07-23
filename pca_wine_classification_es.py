import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
import plotly.graph_objects as go



# Title
st.title('PCA applied to Wine dataset')

# load wine dataset
dataset = load_wine()
# Prepare explanatory variable as DataFrame in pandas
df = pd.DataFrame(dataset.data)
# Assign the names of explanatory variables
df.columns = dataset.feature_names
# Add the target variable(house prices),
# where its column name is set "target".
df["target"] = dataset.target

# Show the table data when checkbox is ON.
if st.checkbox('Show the dataset as table data'):
    st.dataframe(df)

# Prepare the explanatory and target variables
x = df.drop(columns=['target'])
y = df['target']
# Standardization
scaler = StandardScaler()
x_std = scaler.fit_transform(x)


# PCA: principal components analysis

# Number of principal components
st.sidebar.markdown(
    r"""
    ### Select the number of principal components to include in the result
    Note: The number is nonnegative integer.
    """
    )
num_pca = st.sidebar.number_input(
                            'The minimum value is an integer of 3 or more.',
                            value=3, 
                            step=1,
                            min_value=3)
# Perform PCA
pca = PCA(n_components=num_pca)
x_pca = pca.fit_transform(x_std)

st.sidebar.markdown(
    r"""
    ### Select the principal components to plot
    ex. Choose '1' for PCA 1
    """
    )
# Index of PCA, e.g. 1 for PCA 1, 2 for PCA 2, etc..
idx_x_pca = st.sidebar.selectbox("x axis is the principal component of ", np.arange(1, num_pca+1), 0)
idx_y_pca = st.sidebar.selectbox("y axis is the principal component of ", np.arange(1, num_pca+1), 1)
idx_z_pca = st.sidebar.selectbox("z axis is the principal component of ", np.arange(1, num_pca+1), 2)

# Axis label
x_lbl, y_lbl, z_lbl = f"PCA {idx_x_pca}", f"PCA {idx_y_pca}", f"PCA {idx_z_pca}"
# data to plot
x_plot, y_plot, z_plot = x_pca[:,idx_x_pca-1], x_pca[:,idx_y_pca-1], x_pca[:,idx_z_pca-1]

# Create an object for 3d scatter
trace1 = go.Scatter3d(
    x=x_plot, y=y_plot, z=z_plot,
    mode='markers',
    marker=dict(
        size=5,
        color=y,
        # colorscale='Viridis'
        )
)
# Create an object for graph layout
fig = go.Figure(data=[trace1])
fig.update_layout(scene = dict(
                    xaxis_title = x_lbl,
                    yaxis_title = y_lbl,
                    zaxis_title = z_lbl),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10),
                    )
"""### 3d plot of the PCA result by plotly"""
# Plot on the dashboard on streamlit
# fig.write_html("./3dplot.html")
st.plotly_chart(fig, use_container_width=True)


