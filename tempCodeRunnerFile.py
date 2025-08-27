
import numpy as np
import pandas as pd
import plotly.express as px
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator
from quantum_tools import simulate_circuit, get_single_qubit_dm, dm_to_bloch, import_qasm_file, parse_qasm_string
from visualizer import bloch_plot, circuit_diagram_plot
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import base64

# Load custom CSS
def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Quantum State Visualizer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom styling
load_css()

def _figure_to_rl_image(fig, doc_width, target_height=None):
    fig.update_layout(width=900, height=500)
    img_bytes = fig.to_image(format="png", scale=2)
    original_width = 900
    original_height = 500
    scaled_width = doc_width
    scaled_height = (doc_width / original_width) * original_height if target_height is None else target_height
    return RLImage(BytesIO(img_bytes), width=scaled_width, height=scaled_height)

def _format_complex(val: complex) -> str:
    if abs(val) < 1e-10:
        return "0"
    if abs(val.imag) < 1e-10:
        return f"{val.real:.3f}"
    if abs(val.real) < 1e-10:
        return f"{val.imag:.3f}j"
    return f"{val.real:.3f}{val.imag:+.3f}j"

def generate_pdf_report(qc: QuantumCircuit) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    content_width = doc.width

    story.append(Paragraph("Quantum State Visualizer - Report", styles['Title']))
    story.append(Paragraph("Circuit summary and visualizations", styles['Italic']))
    story.append(Spacer(1, 0.2 * inch))

    # Basic circuit info
    story.append(Paragraph(f"Qubits: {qc.num_qubits} | Classical bits: {qc.num_clbits} | Depth: {qc.depth()}", styles['Normal']))
    gate_counts = qc.count_ops()
    if gate_counts:
        counts_text = ", ".join([f"{g}: {c}" for g, c in gate_counts.items()])
        story.append(Paragraph(f"Gate counts: {counts_text}", styles['Normal']))
    story.append(Spacer(1, 0.15 * inch))

    # Circuit diagram
    try:
        circuit_fig = circuit_diagram_plot(qc, "Circuit Diagram")
        story.append(Paragraph("Circuit Diagram", styles['Heading2']))
        story.append(_figure_to_rl_image(circuit_fig, content_width))
        story.append(Spacer(1, 0.2 * inch))
    except Exception:
        pass

    # Export QASM text
    try:
        qasm_text = f"""OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\nqreg q[{qc.num_qubits}];\ncreg c[{qc.num_clbits}];\n\n"""
        for instruction in qc.data:
            gate_name = instruction.operation.name
            qubits = instruction.qubits
            try:
                qubit_indices = [qc.find_bit(q)[0] for q in qubits]
            except Exception:
                qubit_indices = [0, 1]
            if gate_name == 'h':
                qasm_text += f"h q[{qubit_indices[0]}];\n"
            elif gate_name == 'x':
                qasm_text += f"x q[{qubit_indices[0]}];\n"
            elif gate_name == 'y':
                qasm_text += f"y q[{qubit_indices[0]}];\n"
            elif gate_name == 'z':
                qasm_text += f"z q[{qubit_indices[0]}];\n"
            elif gate_name == 'cx':
                qasm_text += f"cx q[{qubit_indices[0]}],q[{qubit_indices[1]}];\n"
            elif gate_name == 'rx':
                angle = instruction.operation.params[0]
                qasm_text += f"rx({angle:.6f}) q[{qubit_indices[0]}];\n"
            elif gate_name == 'ry':
                angle = instruction.operation.params[0]
                qasm_text += f"ry({angle:.6f}) q[{qubit_indices[0]}];\n"
            elif gate_name == 'rz':
                angle = instruction.operation.params[0]
                qasm_text += f"rz({angle:.6f}) q[{qubit_indices[0]}];\n"
            elif gate_name == 'cy':
                qasm_text += f"cy q[{qubit_indices[0]}],q[{qubit_indices[1]}];\n"
            elif gate_name == 'cz':
                qasm_text += f"cz q[{qubit_indices[0]}],q[{qubit_indices[1]}];\n"
        story.append(Paragraph("QASM", styles['Heading2']))
        story.append(Paragraph(f"<pre>{qasm_text}</pre>", styles['Code']))
    except Exception:
        pass

    story.append(PageBreak())

    # Simulation and state data
    rho = simulate_circuit(qc)
    state_vector = rho.data.diagonal() if rho.num_qubits == 1 else rho.data.flatten()
    probabilities = np.abs(state_vector) ** 2
    formatted_vector = [
        _format_complex(val) for val in state_vector
    ]
    prob_formatted = [f"{p:.6f}" for p in probabilities]

    story.append(Paragraph("State Vector |ψ⟩", styles['Heading2']))
    story.append(Paragraph("[" + ", ".join(formatted_vector) + "]", styles['Code']))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("State Probabilities", styles['Heading2']))
    story.append(Paragraph("[" + ", ".join(prob_formatted) + "]", styles['Code']))

    purity = np.trace(rho.data @ rho.data)
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"State Purity: {purity:.4f}", styles['Normal']))

    # Bloch spheres
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Bloch Sphere Representations", styles['Heading2']))
    for i in range(qc.num_qubits):
        try:
            dm = get_single_qubit_dm(rho, i, qc.num_qubits)
            bloch = dm_to_bloch(dm)
            fig = bloch_plot(bloch, title=f"Qubit {i}")
            story.append(Paragraph(f"Qubit {i}", styles['Heading3']))
            story.append(_figure_to_rl_image(fig, content_width))
        except Exception:
            continue

    story.append(PageBreak())

    # Density matrix text (for small systems)
    dm_data = rho.data
    story.append(Paragraph("Density Matrix", styles['Heading2']))
    if dm_data.size <= 16:
        dm_formatted_rows = []
        for row in dm_data:
            row_formatted = [_format_complex(val) for val in row]
            dm_formatted_rows.append("[" + ", ".join(row_formatted) + "]")
        for idx, row_str in enumerate(dm_formatted_rows):
            story.append(Paragraph(f"Row {idx}: {row_str}", styles['Code']))
    else:
        subset = dm_data[:4, :4]
        for i in range(min(4, subset.shape[0])):
            row_vals = [
                f"{val:.3f}" if abs(val.imag) < 1e-10 else f"{val:.3f}"
                for val in subset[i, :4]
            ]
            story.append(Paragraph(f"Row {i}: [" + ", ".join(row_vals) + "]", styles['Code']))
        story.append(Paragraph(f"Matrix size: {dm_data.shape[0]}x{dm_data.shape[1]}. Showing first 4x4 block.", styles['Normal']))

    # Heatmaps
    try:
        prob_matrix = np.abs(rho.data) ** 2
        n_states = 2**qc.num_qubits
        state_labels = [f"|{format(i, f'0{qc.num_qubits}b')}⟩" for i in range(n_states)]
        prob_df = pd.DataFrame(prob_matrix, index=state_labels, columns=state_labels)
        fig_prob = px.imshow(prob_df, title="Probability Matrix |⟨i|ρ|j⟩|²", labels=dict(x="Final State |j⟩", y="Initial State |i⟩"), color_continuous_scale="Blues", aspect="auto", text_auto=True)
        fig_prob.update_layout(height=500)
        fig_prob.update_traces(texttemplate="%{z:.4f}", textfont={"size": 10})
        story.append(PageBreak())
        story.append(Paragraph("Probability Matrix Heatmap", styles['Heading2']))
        story.append(_figure_to_rl_image(fig_prob, content_width))

        real_matrix = np.real(rho.data)
        real_df = pd.DataFrame(real_matrix, index=state_labels, columns=state_labels)
        fig_real = px.imshow(real_df, title="Real Part Re(⟨i|ρ|j⟩)", labels=dict(x="Final State |j⟩", y="Initial State |i⟩"), color_continuous_scale="RdBu_r", aspect="auto", text_auto=True)
        fig_real.update_layout(height=450)
        fig_real.update_traces(texttemplate="%{z:.4f}", textfont={"size": 10})
        story.append(Spacer(1, 0.2 * inch))
        story.append(_figure_to_rl_image(fig_real, content_width))

        imag_matrix = np.imag(rho.data)
        imag_df = pd.DataFrame(imag_matrix, index=state_labels, columns=state_labels)
        fig_imag = px.imshow(imag_df, title="Imaginary Part Im(⟨i|ρ|j⟩)", labels=dict(x="Final State |j⟩", y="Initial State |i⟩"), color_continuous_scale="RdBu_r", aspect="auto", text_auto=True)
        fig_imag.update_layout(height=450)
        fig_imag.update_traces(texttemplate="%{z:.4f}", textfont={"size": 10})
        story.append(Spacer(1, 0.2 * inch))
        story.append(_figure_to_rl_image(fig_imag, content_width))
    except Exception:
        pass

    doc.build(story)
    pdf_value = buffer.getvalue()
    buffer.close()
    return pdf_value

# Fixed top navigation bar
st.markdown("""
<div class="navbar">
  <div class="navbar-inner">
    <div style="display:flex;align-items:center;gap:0.5rem;">
      <span style="font-weight:800;color:#1F2937;">🔮 QSV</span>
    </div>
    <div class="nav-links">
      <a href="#build" >Build</a>
      <a href="#visualize" >Visualize</a>
      <a href="#analyze" >Analyze</a>
      <a href="#examples" >Examples</a>
    </div>
  </div>
</div>
<div class="page-offset"></div>
""", unsafe_allow_html=True)

# Header with enhanced styling
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="background: linear-gradient(135deg, #6D5EF3 0%, #A78BFA 50%, #EC4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 3.25rem; font-weight: 800; margin-bottom: 0.5rem; letter-spacing: -0.02em;">
        🔮 Quantum State Visualizer
    </h1>
    <p style="color: #6B7280; font-size: 1.15rem; margin-bottom: 2rem;">
        Explore quantum circuits and visualize quantum states with modern design
    </p>
</div>
""", unsafe_allow_html=True)

# Decorative divider
st.markdown("---")

# Initialize session state for circuit
if 'circuit' not in st.session_state:
    st.session_state.circuit = QuantumCircuit(2, 2)

# Anchors for navigation
st.markdown('<div id="build" class="anchor-offset"></div>', unsafe_allow_html=True)

# Enhanced Sidebar with Typeform-inspired design
st.sidebar.markdown("""
<div style="background: #F5F5F5; padding: 1.25rem; border-radius: 8px; margin-bottom: 1.5rem; color: #111111; border: 1px solid #E5E7EB;">
    <h3 style="color: #111111; margin: 0; font-size: 1.25rem; font-weight: 600;">
        ⚙️ Circuit Configuration
    </h3>
    <p style="color: #6B7280; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
        Build and configure your quantum circuits
    </p>
</div>
""", unsafe_allow_html=True)

# Top-right toolbar with Download PDF button
with st.container():
    toolbar_left, toolbar_right = st.columns([8, 2])
    with toolbar_right:
        try:
            _toolbar_circuit = st.session_state.get('circuit', None)
            if _toolbar_circuit is not None and _toolbar_circuit.num_qubits > 0 and len(_toolbar_circuit.data) > 0:
                _pdf_bytes_toolbar = generate_pdf_report(_toolbar_circuit)
                st.download_button(
                    label="Download PDF",
                    data=_pdf_bytes_toolbar,
                    file_name="quantum_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf_top"
                )
            else:
                st.button("Download PDF", disabled=True, help="Upload/build a circuit first", use_container_width=True)
        except Exception:
            st.button("Download PDF", disabled=True, help="Simulate the circuit first", use_container_width=True)

# QASM File Upload Section
st.sidebar.markdown("""
<div style="background: #F7FAFC; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #E2E8F0;">
    <h4 style="color: #5B52A5; margin: 0 0 1rem 0; font-size: 1.1rem; font-weight: 600;">
        📁 Import QASM File
    </h4>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Choose a QASM file", type=['qasm', 'txt'])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        qasm_content = uploaded_file.read().decode("utf-8")
        st.sidebar.success("QASM file loaded successfully!")
        
        # Parse QASM and update circuit
        imported_circuit = parse_qasm_string(qasm_content)
        st.session_state.circuit = imported_circuit
        
        # Display QASM content
        with st.sidebar.expander("View QASM Content"):
            st.code(qasm_content, language="text")
            
    except Exception as e:
        st.sidebar.error(f"Error parsing QASM file: {str(e)}")

# Manual Circuit Building Section
st.sidebar.markdown("""
<div style="background: #F7FAFC; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #E2E8F0;">
    <h4 style="color: #5B52A5; margin: 0 0 1rem 0; font-size: 1.1rem; font-weight: 600;">
        🔧 Build Circuit Manually
    </h4>
    <p style="color: #718096; margin: 0; font-size: 0.9rem;">
        Configure circuit parameters and add quantum gates
    </p>
</div>
""", unsafe_allow_html=True)

# Circuit Parameters
st.sidebar.markdown("""
<div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #E2E8F0;">
    <h5 style="color: #2D3748; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 500;">
        Circuit Parameters
    </h5>
</div>
""", unsafe_allow_html=True)

num_qubits = st.sidebar.slider("Number of Qubits", 1, 5, st.session_state.circuit.num_qubits)
num_classical_bits = st.sidebar.slider("Number of Classical Bits", 0, 5, st.session_state.circuit.num_clbits)

# Update circuit if dimensions changed
if (st.session_state.circuit.num_qubits != num_qubits or 
    st.session_state.circuit.num_clbits != num_classical_bits):
    st.session_state.circuit = QuantumCircuit(num_qubits, num_classical_bits)

# Gate Selection Section
st.sidebar.markdown("""
<div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #E2E8F0;">
    <h5 style="color: #2D3748; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 500;">
        🎯 Gate Selection
    </h5>
</div>
""", unsafe_allow_html=True)

target_qubit = st.sidebar.selectbox("Target Qubit", range(num_qubits))

# Pauli Gates Section
st.sidebar.markdown("""
<div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #E2E8F0;">
    <h6 style="color: #5B52A5; margin: 0 0 0.5rem 0; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
        Pauli Gates
    </h6>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.sidebar.columns(4)
with col1:
    if st.button("H"):
        st.session_state.circuit.h(target_qubit)
        st.rerun()
with col2:
    if st.button("X"):
        st.session_state.circuit.x(target_qubit)
        st.rerun()
with col3:
    if st.button("Y"):
        st.session_state.circuit.y(target_qubit)
        st.rerun()
with col4:
    if st.button("Z"):
        st.session_state.circuit.z(target_qubit)
        st.rerun()

# Rotation Gates Section
st.sidebar.markdown("""
<div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #E2E8F0;">
    <h6 style="color: #5B52A5; margin: 0 0 0.5rem 0; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
        Rotation Gates
    </h6>
</div>
""", unsafe_allow_html=True)

angle = st.sidebar.slider("Rotation Angle (π)", 0.0, 2.0, 0.5, 0.1)

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("RX"):
        st.session_state.circuit.rx(angle * np.pi, target_qubit)
        st.rerun()
with col2:
    if st.button("RY"):
        st.session_state.circuit.ry(angle * np.pi, target_qubit)
        st.rerun()
with col3:
    if st.button("RZ"):
        st.session_state.circuit.rz(angle * np.pi, target_qubit)
        st.rerun()

# Two Qubit Gates Section
if num_qubits > 1:
    st.sidebar.markdown("""
    <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #E2E8F0;">
        <h6 style="color: #5B52A5; margin: 0 0 0.5rem 0; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
            Two Qubit Gates
        </h6>
    </div>
    """, unsafe_allow_html=True)
    
    control_qubit = st.sidebar.selectbox("Control Qubit", range(num_qubits))
    if control_qubit != target_qubit:
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            if st.button("CX"):
                st.session_state.circuit.cx(control_qubit, target_qubit)
                st.rerun()
        with col2:
            if st.button("CY"):
                st.session_state.circuit.cy(control_qubit, target_qubit)
                st.rerun()
        with col3:
            if st.button("CZ"):
                st.session_state.circuit.cz(control_qubit, target_qubit)
                st.rerun()

# Special Gates Section
st.sidebar.markdown("""
<div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #E2E8F0;">
    <h6 style="color: #5B52A5; margin: 0 0 0.5rem 0; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
        Special Gates
    </h6>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("S"):
        st.session_state.circuit.s(target_qubit)
        st.rerun()
with col2:
    if st.button("T"):
        st.session_state.circuit.t(target_qubit)
        st.rerun()
with col3:
    if st.button("S†"):
        st.session_state.circuit.sdg(target_qubit)
        st.rerun()

# Circuit Control Section
st.sidebar.markdown("""
<div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #E2E8F0;">
    <h6 style="color: #5B52A5; margin: 0 0 0.5rem 0; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
        Circuit Control
    </h6>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("Reset Circuit"):
        st.session_state.circuit = QuantumCircuit(num_qubits, num_classical_bits)
        st.rerun()
with col2:
    if st.button("Clear"):
        st.session_state.circuit = QuantumCircuit(2, 2)
        st.rerun()
with col3:
    if st.button("Random Circuit"):
        # Generate a random circuit for testing
        import random
        st.session_state.circuit = QuantumCircuit(num_qubits, num_classical_bits)
        for _ in range(random.randint(3, 8)):
            gate = random.choice(['h', 'x', 'y', 'z', 'rx', 'ry', 'rz'])
            qubit = random.randint(0, num_qubits-1)
            if gate in ['rx', 'ry', 'rz']:
                angle = random.uniform(0, 2*np.pi)
                getattr(st.session_state.circuit, gate)(angle, qubit)
            else:
                getattr(st.session_state.circuit, gate)(qubit)
        if num_qubits > 1:
            for _ in range(random.randint(1, 3)):
                ctrl = random.randint(0, num_qubits-1)
                target = random.randint(0, num_qubits-1)
                if ctrl != target:
                    st.session_state.circuit.cx(ctrl, target)
        st.rerun()

# Main content area with enhanced styling
st.markdown('<div id="visualize" class="anchor-offset"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="background: linear-gradient(135deg, #FAFAFB 0%, #FFFFFF 60%); padding: 1.25rem 1.5rem; border-radius: 16px; margin: 1.25rem 0; color: #1F2937; border: 1px solid #E5E7EB; box-shadow: 0 6px 18px rgba(17,24,39,0.06);">
    <h2 style="color: #111111; margin: 0; font-size: 1.5rem; font-weight: 600;">
        🚀 Circuit Visualization & Analysis
    </h2>
    <p style="color: #6B7280; margin: 0.5rem 0 0 0; font-size: 1rem;">
        Explore your quantum circuit and analyze its properties
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div style="background: #F7FAFC; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #E2E8F0;">
        <h3 style="color: #5B52A5; margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 600;">
            📊 Quantum Circuit
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    qc = st.session_state.circuit
    if qc.num_qubits > 0:
        # Display circuit diagram
        circuit_fig = circuit_diagram_plot(qc, "Circuit Diagram")
        st.plotly_chart(circuit_fig, use_container_width=True)
        
        # Also show text representation
        st.write("**Circuit Text Representation:**")
        st.code(qc.draw(output='text'), language="text")
        
        st.write("**Circuit Statistics:**")
        st.write(f"• Qubits: {qc.num_qubits}")
        st.write(f"• Classical bits: {qc.num_clbits}")
        st.write(f"• Circuit depth: {qc.depth()}")
        st.write(f"• Total gates: {qc.num_nonlocal_gates() + qc.num_qubits}")
        
        # Gate counts
        gate_counts = qc.count_ops()
        if gate_counts:
            st.write("**Gate counts:**")
            for gate, count in gate_counts.items():
                st.write(f"  - {gate}: {count}")
    else:
        st.info("Add gates to build your circuit!")

with col2:
    st.markdown("""
    <div style="background: #F7FAFC; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #E2E8F0;">
        <h3 style="color: #5B52A5; margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 600;">
            🔍 Circuit Analysis
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    if qc.num_qubits > 0:
        # Export QASM
        st.write("**Export QASM:**")
        try:
            # Create QASM manually for better compatibility
            qasm_text = f"""OPENQASM 2.0;
include "qelib1.inc";

qreg q[{qc.num_qubits}];
creg c[{qc.num_clbits}];

"""
            # Add gates based on circuit data using new Qiskit API
            for instruction in qc.data:
                gate_name = instruction.operation.name
                qubits = instruction.qubits
                clbits = instruction.clbits
                
                # Get qubit indices using the correct method
                try:
                    qubit_indices = [qc.find_bit(q)[0] for q in qubits]
                except:
                    # Fallback method
                    qubit_indices = [0, 1]  # Default fallback
                
                if gate_name == 'h':
                    qasm_text += f"h q[{qubit_indices[0]}];\n"
                elif gate_name == 'x':
                    qasm_text += f"x q[{qubit_indices[0]}];\n"
                elif gate_name == 'y':
                    qasm_text += f"y q[{qubit_indices[0]}];\n"
                elif gate_name == 'z':
                    qasm_text += f"z q[{qubit_indices[0]}];\n"
                elif gate_name == 'cx':
                    qasm_text += f"cx q[{qubit_indices[0]}],q[{qubit_indices[1]}];\n"
                elif gate_name == 'rx':
                    angle = instruction.operation.params[0]
                    qasm_text += f"rx({angle:.6f}) q[{qubit_indices[0]}];\n"
                elif gate_name == 'ry':
                    angle = instruction.operation.params[0]
                    qasm_text += f"ry({angle:.6f}) q[{qubit_indices[0]}];\n"
                elif gate_name == 'rz':
                    angle = instruction.operation.params[0]
                    qasm_text += f"rz({angle:.6f}) q[{qubit_indices[0]}];\n"
                elif gate_name == 'cy':
                    qasm_text += f"cy q[{qubit_indices[0]}],q[{qubit_indices[1]}];\n"
                elif gate_name == 'cz':
                    qasm_text += f"cz q[{qubit_indices[0]}],q[{qubit_indices[1]}];\n"
        
        except Exception as e:
            qasm_text = f"# Error generating QASM: {str(e)}\n# Circuit has {qc.num_qubits} qubits and {qc.num_clbits} classical bits"
        
        st.code(qasm_text, language="text")
        
        # Download button for QASM
        st.download_button(
            label="Download QASM File",
            data=qasm_text,
            file_name="quantum_circuit.qasm",
            mime="text/plain"
        )
        
        # Circuit properties
        st.write("**Circuit Properties:**")
        st.write(f"• **Width**: {qc.width()}")
        st.write(f"• **Size**: {qc.size()}")
        st.write(f"• **Depth**: {qc.depth()}")
        
        # Show circuit data structure
        with st.expander("View Circuit Data Structure"):
            st.write("**Instructions:**")
            for i, instruction in enumerate(qc.data):
                gate_name = instruction.operation.name
                qubits = instruction.qubits
                clbits = instruction.clbits
                
                # Get qubit indices using the correct method
                try:
                    qubit_indices = [qc.find_bit(q)[0] for q in qubits]
                except:
                    # Fallback method
                    qubit_indices = [0, 1]
                
                st.write(f"{i}: {gate_name} on qubits {qubit_indices}")
                
                if hasattr(instruction.operation, 'params') and instruction.operation.params:
                    st.write(f"   Parameters: {instruction.operation.params}")

# Circuit simulation and visualization
if qc.num_qubits > 0:
    st.markdown('<div id="analyze" class="anchor-offset"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #FAFAFB 0%, #FFFFFF 60%); padding: 1.25rem 1.5rem; border-radius: 16px; margin: 1.25rem 0; color: #1F2937; border: 1px solid #E5E7EB; box-shadow: 0 6px 18px rgba(17,24,39,0.06);">
        <h2 style="color: #111111; margin: 0; font-size: 1.5rem; font-weight: 600;">
            🎯 Quantum State Analysis
        </h2>
        <p style="color: #6B7280; margin: 0.5rem 0 0 0; font-size: 1rem;">
            Explore the quantum state and its properties
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Simulate the circuit
        rho = simulate_circuit(qc)

        # Display state vector
        st.write("**State Vector:**")
        state_vector = rho.data.diagonal() if rho.num_qubits == 1 else rho.data.flatten()
        
        # Format complex numbers for better display
        formatted_vector = []
        for i, val in enumerate(state_vector):
            if abs(val) < 1e-10:  # Very small values
                formatted_vector.append("0")
            elif abs(val.imag) < 1e-10:  # Real numbers
                formatted_vector.append(f"{val.real:.3f}")
            elif abs(val.real) < 1e-10:  # Pure imaginary
                formatted_vector.append(f"{val.imag:.3f}j")
            else:  # Complex numbers
                formatted_vector.append(f"{val.real:.3f}{val.imag:+.3f}j")
        
        # Display in a more readable format
        st.write("**State Vector (|ψ⟩):**")
        st.code("[" + ", ".join(formatted_vector) + "]", language="text")
        
        # Show state probabilities
        st.write("**State Probabilities:**")
        probabilities = np.abs(state_vector) ** 2
        prob_formatted = [f"{p:.6f}" for p in probabilities]
        st.code("[" + ", ".join(prob_formatted) + "]", language="text")
        
        # Show non-zero states
        non_zero_states = [(i, val, prob) for i, (val, prob) in enumerate(zip(state_vector, probabilities)) if abs(val) > 1e-10]
        if non_zero_states:
            st.write("**Non-zero basis states:**")
            for i, val, prob in non_zero_states:
                binary_state = format(i, f'0{qc.num_qubits}b')
                st.write(f"|{binary_state}⟩: amplitude = {val:.6f}, probability = {prob:.6f}")
            
            # Interpret the quantum state
            st.write("**Quantum State Interpretation:**")
            if len(non_zero_states) == 1:
                binary_state = format(non_zero_states[0][0], f'0{qc.num_qubits}b')
                st.success(f"This circuit creates the computational basis state |{binary_state}⟩")
            elif len(non_zero_states) == 2 and abs(abs(non_zero_states[0][1]) - abs(non_zero_states[1][1])) < 1e-10:
                if qc.num_qubits == 2:
                    st.success("This appears to be a Bell state or similar maximally entangled state!")
                else:
                    st.success("This appears to be a maximally entangled state!")
            elif len(non_zero_states) == 4 and all(abs(abs(val) - 0.5) < 1e-10 for _, val, _ in non_zero_states):
                st.success("This appears to be a uniform superposition of four basis states!")
            else:
                st.info(f"This circuit creates a superposition of {len(non_zero_states)} basis states")
        else:
            st.info("Circuit is in initial state |00...0⟩")
        
        # Bloch spheres for each qubit
        st.write("**Bloch Sphere Representations:**")
        
        cols = st.columns(min(3, num_qubits))
        for i in range(num_qubits):
            with cols[i % 3]:
                try:
                    dm = get_single_qubit_dm(rho, i, num_qubits)
                    bloch = dm_to_bloch(dm)
                    fig = bloch_plot(bloch, title=f"Qubit {i}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display Bloch coordinates
                    st.write(f"**Qubit {i} Bloch coordinates:**")
                    st.write(f"X: {bloch[0]:.3f}")
                    st.write(f"Y: {bloch[1]:.3f}")
                    st.write(f"Z: {bloch[2]:.3f}")
                except Exception as e:
                    st.error(f"Error visualizing qubit {i}: {str(e)}")
        
        # Density matrix visualization
        st.write("**Density Matrix:**")
        
        # Format density matrix for better display
        dm_data = rho.data
        if dm_data.size <= 16:  # Only show full matrix for small systems
            st.write("**Full Density Matrix:**")
            # Create a formatted display
            dm_formatted = []
            for row in dm_data:
                row_formatted = []
                for val in row:
                    if abs(val) < 1e-10:
                        row_formatted.append("0")
                    elif abs(val.imag) < 1e-10:
                        row_formatted.append(f"{val.real:.3f}")
                    elif abs(val.real) < 1e-10:
                        row_formatted.append(f"{val.imag:.3f}j")
                    else:
                        row_formatted.append(f"{val.real:.3f}{val.imag:+.3f}j")
                dm_formatted.append(row_formatted)
            
            # Display as a table
            for i, row in enumerate(dm_formatted):
                st.write(f"Row {i}: [{', '.join(row)}]")
        else:
            st.write("**Density Matrix (showing first 4x4 block):**")
            # Show only a subset for large matrices
            subset = dm_data[:4, :4]
            for i in range(min(4, subset.shape[0])):
                row_str = ", ".join([f"{val:.3f}" if abs(val.imag) < 1e-10 else f"{val:.3f}" for val in subset[i, :4]])
                st.write(f"Row {i}: [{row_str}]")
            st.info(f"Matrix size: {dm_data.shape[0]}x{dm_data.shape[1]}. Showing first 4x4 block.")
        
        # Purity calculation
        purity = np.trace(rho.data @ rho.data)
        st.write(f"**State Purity:** {purity:.4f}")
        if abs(purity - 1.0) < 1e-10:
            st.success("Pure state")
        else:
            st.warning("Mixed state")
        
        # Probability Matrix Heatmap
        st.markdown("---")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FAFAFB 0%, #FFFFFF 60%); padding: 1.25rem 1.5rem; border-radius: 16px; margin: 1.25rem 0; color: #1F2937; border: 1px solid #E5E7EB; box-shadow: 0 6px 18px rgba(17,24,39,0.06);">
            <h2 style="color: #111111; margin: 0; font-size: 1.5rem; font-weight: 600;">
                🔥 Probability Matrix Heatmap
            </h2>
            <p style="color: #6B7280; margin: 0.5rem 0 0 0; font-size: 1rem;">
                Visualize the probability distribution and matrix elements
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Clean UI: removed debug/test messages
        
        try:
            # Create probability matrix from density matrix
            prob_matrix = np.abs(rho.data) ** 2
            
            # Create heatmap for probability matrix
            
            # Prepare data for heatmap
            n_states = 2**qc.num_qubits
            state_labels = [f"|{format(i, f'0{qc.num_qubits}b')}⟩" for i in range(n_states)]
            
            # Clean UI: removed extra matrix debug info
            
            # Create DataFrame for probability heatmap
            prob_df = pd.DataFrame(prob_matrix, 
                                  index=state_labels, 
                                  columns=state_labels)
            
            # Probability heatmap
            fig_prob = px.imshow(prob_df, 
                                title="Probability Matrix |⟨i|ρ|j⟩|²",
                                labels=dict(x="Final State |j⟩", y="Initial State |i⟩"),
                                color_continuous_scale="Blues",
                                aspect="auto",
                                text_auto=True)
            fig_prob.update_layout(height=500)
            fig_prob.update_traces(texttemplate="%{z:.4f}", textfont={"size": 10})
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # Real and Imaginary parts heatmaps
            col1, col2 = st.columns(2)
            
            with col1:
                # Real part heatmap
                real_matrix = np.real(rho.data)
                real_df = pd.DataFrame(real_matrix, 
                                      index=state_labels, 
                                      columns=state_labels)
                
                fig_real = px.imshow(real_df, 
                                    title="Real Part Re(⟨i|ρ|j⟩)",
                                    labels=dict(x="Final State |j⟩", y="Initial State |i⟩"),
                                    color_continuous_scale="RdBu_r",
                                    aspect="auto",
                                    text_auto=True)
                fig_real.update_layout(height=400)
                fig_real.update_traces(texttemplate="%{z:.4f}", textfont={"size": 10})
                st.plotly_chart(fig_real, use_container_width=True)
            
            with col2:
                # Imaginary part heatmap
                imag_matrix = np.imag(rho.data)
                imag_df = pd.DataFrame(imag_matrix, 
                                      index=state_labels, 
                                      columns=state_labels)
                
                fig_imag = px.imshow(imag_df, 
                                    title="Imaginary Part Im(⟨i|ρ|j⟩)",
                                    labels=dict(x="Final State |j⟩", y="Initial State |i⟩"),
                                    color_continuous_scale="RdBu_r",
                                    aspect="auto",
                                    text_auto=True)
                fig_imag.update_layout(height=400)
                fig_imag.update_traces(texttemplate="%{z:.4f}", textfont={"size": 10})
                st.plotly_chart(fig_imag, use_container_width=True)
            
            # Matrix statistics
            st.write("**Matrix Statistics:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Max Probability", f"{np.max(prob_matrix):.4f}")
                st.metric("Min Probability", f"{np.min(prob_matrix):.4f}")
            
            with col2:
                st.metric("Max Real Part", f"{np.max(real_matrix):.4f}")
                st.metric("Min Real Part", f"{np.min(real_matrix):.4f}")
            
            with col3:
                st.metric("Max Imag Part", f"{np.max(imag_matrix):.4f}")
                st.metric("Min Imag Part", f"{np.min(imag_matrix):.4f}")
            
            # Matrix summary
            with st.expander("📊 Matrix Summary"):
                st.write("**Probability Matrix (|⟨i|ρ|j⟩|²):**")
                st.write("Shows the probability of measuring state |j⟩ when the system is in state |i⟩")
                
                st.write("**Real Part Matrix (Re(⟨i|ρ|j⟩)):**")
                st.write("Shows the real component of the density matrix elements")
                
                st.write("**Imaginary Part Matrix (Im(⟨i|ρ|j⟩)):**")
                st.write("Shows the imaginary component of the density matrix elements")
                
                st.write("**Color Scale:**")
                st.write("- **Blues**: Higher probabilities are darker blue")
                st.write("- **RdBu_r**: Red = negative, Blue = positive, White = zero")
            
            # Detailed numerical matrices
            st.markdown("---")
            st.subheader("📋 Detailed Numerical Matrices")
            
            # Probability matrix table
            st.write("**Probability Matrix Values:**")
            prob_table = pd.DataFrame(prob_matrix, 
                                    index=state_labels, 
                                    columns=state_labels)
            st.dataframe(prob_table.style.format("{:.6f}"), use_container_width=True)
            
            # Real and imaginary matrices side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Real Part Matrix Values:**")
                real_table = pd.DataFrame(real_matrix, 
                                        index=state_labels, 
                                        columns=state_labels)
                st.dataframe(real_table.style.format("{:.6f}"), use_container_width=True)
            
            with col2:
                st.write("**Imaginary Part Matrix Values:**")
                imag_table = pd.DataFrame(imag_matrix, 
                                        index=state_labels, 
                                        columns=state_labels)
                st.dataframe(imag_table.style.format("{:.6f}"), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error creating heatmaps: {str(e)}")
            st.write("**Raw Data for Debugging:**")
            st.write(f"Rho data type: {type(rho.data)}")
            st.write(f"Rho data shape: {rho.data.shape}")
            st.write(f"Rho data: {rho.data}")
            import traceback
            st.code(traceback.format_exc(), language="text")
        
    except Exception as e:
        st.error(f"Error simulating circuit: {str(e)}")
        st.info("Try simplifying your circuit or check for invalid gate combinations.")

# Combined Examples section with tabs
st.markdown("""
<div style="background: linear-gradient(135deg, #FAFAFB 0%, #FFFFFF 60%); padding: 1.25rem 1.5rem; border-radius: 16px; margin: 1.25rem 0; color: #1F2937; border: 1px solid #E5E7EB; box-shadow: 0 6px 18px rgba(17,24,39,0.06);">
    <h2 style="margin:0; font-size:1.5rem; font-weight:700;">📚 Circuit Examples</h2>
    <p style="color:#6B7280; margin:0.25rem 0 0 0;">Quick presets and advanced examples in one place</p>
</div>
""", unsafe_allow_html=True)

examples_tab1, examples_tab2 = st.tabs(["Quick Presets", "Advanced Examples"])

with examples_tab1:
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    with preset_col1:
        if st.button("H on All"):
            for i in range(num_qubits):
                st.session_state.circuit.h(i)
            st.rerun()
    with preset_col2:
        if st.button("X on All"):
            for i in range(num_qubits):
                st.session_state.circuit.x(i)
            st.rerun()
    with preset_col3:
        if st.button("Y on All"):
            for i in range(num_qubits):
                st.session_state.circuit.y(i)
            st.rerun()
    with preset_col4:
        if st.button("Z on All"):
            for i in range(num_qubits):
                st.session_state.circuit.z(i)
            st.rerun()

with examples_tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Bell State"):
            st.session_state.circuit = QuantumCircuit(2, 2)
            st.session_state.circuit.h(0)
            st.session_state.circuit.cx(0, 1)
            st.rerun()
    with col2:
        if st.button("GHZ State"):
            st.session_state.circuit = QuantumCircuit(3, 3)
            st.session_state.circuit.h(0)
            st.session_state.circuit.cx(0, 1)
            st.session_state.circuit.cx(1, 2)
            st.rerun()
    with col3:
        if st.button("W State"):
            st.session_state.circuit = QuantumCircuit(3, 3)
            st.session_state.circuit.ry(2.094, 0)
            st.session_state.circuit.cx(0, 1)
            st.session_state.circuit.ry(1.571, 1)
            st.session_state.circuit.cx(1, 2)
            st.rerun()

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="background: #2D3748; padding: 2rem; border-radius: 12px; margin-top: 3rem; text-align: center; color: white;">
    <p style="margin: 0; font-size: 1rem; color: #A0AEC0;">
        🔮 <strong>Quantum State Visualizer</strong> - Built with Qiskit, Streamlit, and Plotly
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #718096;">
        Explore the fascinating world of quantum computing with modern design
    </p>
</div>
""", unsafe_allow_html=True)
