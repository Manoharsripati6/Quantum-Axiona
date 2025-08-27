import numpy as np
import plotly.graph_objects as go

def bloch_plot(bloch_vector, title="Bloch Sphere"):
    """Return a Plotly figure for a Bloch vector."""
    x, y, z = bloch_vector
    
    fig = go.Figure()
    
    # Create sphere surface
    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 25)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    x_sphere = np.sin(theta_grid) * np.cos(phi_grid)
    y_sphere = np.sin(theta_grid) * np.sin(phi_grid)
    z_sphere = np.cos(theta_grid)
    
    # Add sphere surface (higher opacity, more visible colorscale)
    fig.add_trace(go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        opacity=0.4,
        showscale=False,
        colorscale='YlGnBu',
        hoverinfo='skip'
    ))
    
    # Add coordinate axes
    # X-axis (red)
    fig.add_trace(go.Scatter3d(
        x=[-1.2, 1.2], y=[0, 0], z=[0, 0],
        mode="lines",
        line=dict(color="red", width=3),
        name="X-axis",
        hoverinfo='skip'
    ))
    
    # Y-axis (green)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-1.2, 1.2], z=[0, 0],
        mode="lines",
        line=dict(color="green", width=3),
        name="Y-axis",
        hoverinfo='skip'
    ))
    
    # Z-axis (blue)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-1.2, 1.2],
        mode="lines",
        line=dict(color="blue", width=3),
        name="Z-axis",
        hoverinfo='skip'
    ))
    
    # Add Bloch vector
    if np.linalg.norm(bloch_vector) > 1e-10:  # Only show if vector is non-zero
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode="lines+markers",
            line=dict(color="purple", width=8),
            marker=dict(size=8, color="purple", symbol="diamond"),
            name="State Vector",
            hovertemplate="<b>State Vector</b><br>" +
                         f"X: {x:.3f}<br>" +
                         f"Y: {y:.3f}<br>" +
                         f"Z: {z:.3f}<br>" +
                         "<extra></extra>"
        ))
    
    # Add origin point
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=5, color="black"),
        name="Origin",
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color="black")
        ),
        scene=dict(
            xaxis=dict(
                range=[-1.2, 1.2],
                title="X",
                tickfont=dict(color="red"),
                gridcolor="lightgray",
                zerolinecolor="lightgray"
            ),
            yaxis=dict(
                range=[-1.2, 1.2],
                title="Y",
                tickfont=dict(color="green"),
                gridcolor="lightgray",
                zerolinecolor="lightgray"
            ),
            zaxis=dict(
                range=[-1.2, 1.2],
                title="Z",
                tickfont=dict(color="blue"),
                gridcolor="lightgray",
                zerolinecolor="lightgray"
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=500
    )
    
    return fig

def state_probability_plot(probabilities, title="State Probabilities"):
    """Create a bar plot of computational basis state probabilities."""
    num_states = len(probabilities)
    state_labels = [f"|{bin(i)[2:].zfill(int(np.log2(num_states)))}⟩" for i in range(num_states)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=state_labels,
            y=probabilities,
            marker_color='lightblue',
            marker_line_color='navy',
            marker_line_width=1,
            opacity=0.8
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Computational Basis States",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1.1]),
        showlegend=False,
        height=400
    )
    
    return fig

def circuit_diagram_plot(circuit, title="Quantum Circuit"):
    """Create a visual representation of the quantum circuit."""
    # Get circuit information
    num_qubits = circuit.num_qubits
    num_clbits = circuit.num_clbits
    depth = circuit.depth()
    
    # Create a simple text-based circuit representation
    fig = go.Figure()
    
    # Add circuit info as text
    fig.add_annotation(
        x=0.5, y=0.9,
        xref="paper", yref="paper",
        text=f"Qubits: {num_qubits} | Classical bits: {num_clbits} | Depth: {depth}",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="lightblue",
        bordercolor="navy",
        borderwidth=2
    )
    
    # Add circuit diagram as text
    circuit_text = circuit.draw(output='text')
    fig.add_annotation(
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        text=f"<pre>{circuit_text}</pre>",
        showarrow=False,
        font=dict(size=10, family="monospace"),
        align="left"
    )
    
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=400,
        showlegend=False
    )
    
    return fig
