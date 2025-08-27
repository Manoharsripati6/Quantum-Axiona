from flask import Flask, request, send_file, make_response, render_template, url_for, session
import io
import base64
import numpy as np
import pandas as pd
import plotly.express as px
from qiskit import QuantumCircuit
from quantum_tools import simulate_circuit, get_single_qubit_dm, dm_to_bloch, parse_qasm_string
from visualizer import bloch_plot, circuit_diagram_plot
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


app = Flask(__name__, template_folder="templates")
app.secret_key = "dev-secret"

# ------------------------ Circuit persistence helpers ------------------------
def _get_qc() -> QuantumCircuit:
    qasm_text = session.get('qc_qasm')
    if qasm_text:
        try:
            return parse_qasm_string(qasm_text)
        except Exception:
            pass
    qc = QuantumCircuit(2, 2)
    session['qc_qasm'] = qc_to_qasm(qc)
    return qc

def _set_qc(qc: QuantumCircuit):
    session['qc_qasm'] = qc_to_qasm(qc)

# ------------------------ Report Utilities ------------------------
def _figure_to_rl_image(fig, doc_width, target_height=None):
    fig.update_layout(width=900, height=500)
    img_bytes = fig.to_image(format="png", scale=2)
    original_width = 900
    original_height = 500
    scaled_width = doc_width
    scaled_height = (doc_width / original_width) * original_height if target_height is None else target_height
    return RLImage(io.BytesIO(img_bytes), width=scaled_width, height=scaled_height)

def _format_complex(val: complex) -> str:
    if abs(val) < 1e-10:
        return "0"
    if abs(val.imag) < 1e-10:
        return f"{val.real:.3f}"
    if abs(val.real) < 1e-10:
        return f"{val.imag:.3f}j"
    return f"{val.real:.3f}{val.imag:+.3f}j"

def generate_pdf_report(qc: QuantumCircuit) -> bytes:

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    content_width = doc.width

    # --- Precompute quantum state and probabilities for all plots ---
    rho = simulate_circuit(qc)
    state_vector = rho.data.diagonal() if rho.num_qubits == 1 else rho.data.flatten()
    probabilities = np.abs(state_vector) ** 2

    # --- Title and Summary ---
    story.append(Paragraph("Quantum Axioma - Report", styles['Title']))
    story.append(Paragraph("Comprehensive quantum circuit analysis and state visualization", styles['Italic']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Qubits: {qc.num_qubits} | Classical bits: {qc.num_clbits} | Depth: {qc.depth()} | Total Gates: {qc.size()}", styles['Normal']))
    gate_counts = qc.count_ops()
    if gate_counts:
        counts_text = ", ".join([f"{g}: {c}" for g, c in gate_counts.items()])
        story.append(Paragraph(f"Gate counts: {counts_text}", styles['Normal']))
    story.append(Spacer(1, 0.15 * inch))

    # --- Section: Circuit Diagram & QASM ---
    story.append(Paragraph("Circuit Diagram & QASM", styles['Heading1']))
    story.append(Spacer(1, 0.1 * inch))
    # Matplotlib Circuit Diagram ONLY
    try:
        fig = qc.draw(output='mpl')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=220)
        plt.close(fig)
        buf.seek(0)
        story.append(Paragraph("Circuit Diagram", styles['Heading2']))
        story.append(RLImage(buf, width=content_width, height=content_width*0.4))
        story.append(Spacer(1, 0.2 * inch))
    except Exception as e:
        print(f"[PDF] Error generating circuit diagram: {e}")
    # --- Section: Probability Bar Plot ---
    story.append(Paragraph("State Probabilities", styles['Heading1']))
    story.append(Spacer(1, 0.1 * inch))
    try:
        import plotly.graph_objects as go
        n_states = 2 ** qc.num_qubits
        state_labels = [f"|{format(i, f'0{qc.num_qubits}b')}⟩" for i in range(n_states)]
        max_prob = float(np.max(probabilities))
        if max_prob < 1e-10:
            y_max = 1.0
        else:
            import math
            y_max = math.ceil(max_prob * 10) / 10.0
            if y_max < max_prob:
                y_max = max_prob
            if y_max < 1.0:
                y_max = round(y_max + 0.1, 1)
        fig_prob = go.Figure(
            data=[go.Bar(x=state_labels, y=probabilities, marker_color='royalblue')],
            layout=go.Layout(
                title="State Probabilities",
                xaxis_title="State",
                yaxis_title="Probability",
                yaxis=dict(range=[0, y_max])
            )
        )
        buf = io.BytesIO()
        fig_prob.write_image(buf, format='png', scale=2)
        buf.seek(0)
        story.append(RLImage(buf, width=content_width, height=content_width*0.4))
        story.append(Spacer(1, 0.2 * inch))
    except Exception as e:
        print(f"[PDF] Error generating probability bar plot: {e}")
    # QASM code
    try:
        qasm_text = qc_to_qasm(qc)
        story.append(Paragraph("QASM Code", styles['Heading2']))
        story.append(Paragraph(f"<pre>{qasm_text}</pre>", styles['Code']))
        story.append(Spacer(1, 0.2 * inch))
    except Exception:
        pass
    story.append(PageBreak())

    # --- Section: Bloch Spheres ---
    story.append(Paragraph("Bloch Spheres", styles['Heading1']))
    story.append(Spacer(1, 0.1 * inch))

    # Quantum state analysis
    formatted_vector = [_format_complex(val) for val in state_vector]
    prob_formatted = [f"{p:.6f}" for p in probabilities]

    story.append(Paragraph("Quantum State Analysis", styles['Heading1']))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("State Vector |ψ⟩", styles['Heading2']))
    story.append(Paragraph("[" + ", ".join(formatted_vector) + "]", styles['Code']))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("State Probabilities", styles['Heading2']))
    story.append(Paragraph("[" + ", ".join(prob_formatted) + "]", styles['Code']))
    story.append(Spacer(1, 0.2 * inch))

    # Add probability heatmap (plotly)
    try:
        import pandas as pd
        import plotly.express as px
        n_states = 2**qc.num_qubits
        state_labels = [f"|{format(i, f'0{qc.num_qubits}b')}⟩" for i in range(n_states)]
        prob_df = pd.DataFrame(np.abs(rho.data) ** 2, index=state_labels, columns=state_labels)
        heatmap = px.imshow(prob_df, title="Probability Matrix |⟨i|ρ|j⟩|²", color_continuous_scale="Blues", aspect="auto", text_auto=True)
        heatmap.update_layout(width=600, height=500)
        buf = io.BytesIO()
        heatmap.write_image(buf, format='png', scale=2)
        buf.seek(0)
        story.append(Paragraph("Probability Heatmap", styles['Heading2']))
        story.append(RLImage(buf, width=content_width, height=content_width*0.6))
        story.append(Spacer(1, 0.2 * inch))
    except Exception as e:
        print(f"[PDF] Error generating probability heatmap: {e}")

    # Add real/imaginary heatmaps (plotly)
    try:
        real_df = pd.DataFrame(np.real(rho.data), index=state_labels, columns=state_labels)
        real_map = px.imshow(real_df, title="Real Part Re(⟨i|ρ|j⟩)", color_continuous_scale="RdBu_r", aspect="auto", text_auto=True)
        real_map.update_layout(width=600, height=400)
        buf = io.BytesIO()
        real_map.write_image(buf, format='png', scale=2)
        buf.seek(0)
        story.append(Paragraph("Real Part Heatmap", styles['Heading2']))
        story.append(RLImage(buf, width=content_width, height=content_width*0.5))
        story.append(Spacer(1, 0.1 * inch))
    except Exception as e:
        print(f"[PDF] Error generating real part heatmap: {e}")
    try:
        imag_df = pd.DataFrame(np.imag(rho.data), index=state_labels, columns=state_labels)
        imag_map = px.imshow(imag_df, title="Imaginary Part Im(⟨i|ρ|j⟩)", color_continuous_scale="RdBu_r", aspect="auto", text_auto=True)
        imag_map.update_layout(width=600, height=400)
        buf = io.BytesIO()
        imag_map.write_image(buf, format='png', scale=2)
        buf.seek(0)
        story.append(Paragraph("Imaginary Part Heatmap", styles['Heading2']))
        story.append(RLImage(buf, width=content_width, height=content_width*0.5))
        story.append(Spacer(1, 0.2 * inch))
    except Exception as e:
        print(f"[PDF] Error generating imaginary part heatmap: {e}")

    # Add Bloch spheres (Plotly, matching UI)
    if qc.num_qubits <= 3:
        story.append(Paragraph("Bloch Spheres", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))
        bloch_img_size = min(content_width*0.5, 3*inch)  # Square size, max 3 inches
        for i in range(qc.num_qubits):
            try:
                dm = get_single_qubit_dm(rho, i, qc.num_qubits)
                bloch = dm_to_bloch(dm)
                fig_b = bloch_plot(bloch, title=f"Qubit {i}")
                fig_b.update_layout(width=400, height=400, margin=dict(l=0, r=0, t=40, b=0))
                buf = io.BytesIO()
                fig_b.write_image(buf, format='png', scale=2)
                buf.seek(0)
                story.append(RLImage(buf, width=bloch_img_size, height=bloch_img_size))
                story.append(Spacer(1, 0.1 * inch))
            except Exception as e:
                print(f"[PDF] Error generating Plotly Bloch sphere (UI match) for qubit {i}: {e}")
                continue

    # Detailed state-by-state analysis
    story.append(Paragraph("Detailed State Analysis", styles['Heading2']))
    story.append(Spacer(1, 0.1 * inch))

    n_states = 2**qc.num_qubits
    state_labels = [format(i, f'0{qc.num_qubits}b') for i in range(n_states)]
    
    for i, (state_label, amplitude, prob) in enumerate(zip(state_labels, state_vector, probabilities)):
        if abs(amplitude) > 1e-10:  # Only show non-zero states
            story.append(Paragraph(f"State |{state_label}⟩:", styles['Heading3']))
            
            # Amplitude details
            real_part = np.real(amplitude)
            imag_part = np.imag(amplitude)
            magnitude = abs(amplitude)
            phase = np.angle(amplitude, deg=True)
            
            details = f"  • Amplitude: {_format_complex(amplitude)}"
            details += f"<br/>  • Magnitude: {magnitude:.6f}"
            details += f"<br/>  • Phase: {phase:.2f}°"
            details += f"<br/>  • Probability: {prob:.6f} ({prob*100:.2f}%)"
            
            if abs(real_part) > 1e-10:
                details += f"<br/>  • Real part: {real_part:.6f}"
            if abs(imag_part) > 1e-10:
                details += f"<br/>  • Imaginary part: {imag_part:.6f}"
            
            story.append(Paragraph(details, styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))

    story.append(Spacer(1, 0.2 * inch))

    # Purity and interpretation
    purity = float(np.real_if_close(np.trace(rho.data @ rho.data)))
    story.append(Paragraph("State Properties", styles['Heading2']))
    story.append(Paragraph(f"Purity: {purity:.6f} ({'Pure state' if purity > 0.999 else 'Mixed state'})", styles['Normal']))
    
    # State interpretation
    non_zero_states = [(i, v, p) for i, (v, p) in enumerate(zip(state_vector, probabilities)) if abs(v) > 1e-10]
    if len(non_zero_states) == 1:
        b = format(non_zero_states[0][0], f'0{qc.num_qubits}b')
        interpretation = f"Computational basis state |{b}⟩"
    elif len(non_zero_states) == 2 and abs(abs(non_zero_states[0][1]) - abs(non_zero_states[1][1])) < 1e-10:
        interpretation = "Maximally entangled-like superposition"
    elif len(non_zero_states) == 4 and all(abs(abs(v) - 0.5) < 1e-10 for _, v, _ in non_zero_states):
        interpretation = "Uniform superposition of four basis states"
    else:
        interpretation = f"Superposition of {len(non_zero_states)} states"
    
    story.append(Paragraph(f"Interpretation: {interpretation}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Bloch sphere information for each qubit
    if qc.num_qubits <= 3:  # Only for reasonable number of qubits
        story.append(Paragraph("Individual Qubit Analysis", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))
        
        for i in range(qc.num_qubits):
            try:
                dm = get_single_qubit_dm(rho, i, qc.num_qubits)
                bloch = dm_to_bloch(dm)
                story.append(Paragraph(f"Qubit {i} Bloch coordinates:", styles['Heading3']))
                story.append(Paragraph(f"  • X: {bloch[0]:.6f}", styles['Normal']))
                story.append(Paragraph(f"  • Y: {bloch[1]:.6f}", styles['Normal']))
                story.append(Paragraph(f"  • Z: {bloch[2]:.6f}", styles['Normal']))
                story.append(Spacer(1, 0.1 * inch))
            except Exception:
                continue

    # Density matrix (first 4x4 for readability)
    try:
        dm = simulate_circuit(qc).data
        rows = min(4, dm.shape[0])
        cols = min(4, dm.shape[1])
        story.append(Paragraph("Density Matrix (first 4x4)", styles['Heading2']))
        for i in range(rows):
            row_vals = [f"{np.real(dm[i,j]):.3f}{np.imag(dm[i,j]):+ .3f}j" if abs(np.imag(dm[i,j]))>1e-10 else f"{np.real(dm[i,j]):.3f}" for j in range(cols)]
            story.append(Paragraph("[" + ", ".join(row_vals) + "]", styles['Code']))
        story.append(Spacer(1, 0.2 * inch))
    except Exception:
        pass

    # Circuit statistics
    story.append(Paragraph("Circuit Statistics", styles['Heading2']))
    story.append(Paragraph(f"• Total operations: {qc.size()}", styles['Normal']))
    story.append(Paragraph(f"• Circuit depth: {qc.depth()}", styles['Normal']))
    story.append(Paragraph(f"• Width: {qc.num_qubits} qubits", styles['Normal']))
    if qc.num_clbits > 0:
        story.append(Paragraph(f"• Classical bits: {qc.num_clbits}", styles['Normal']))

    doc.build(story)
    pdf_value = buffer.getvalue()
    buffer.close()
    return pdf_value

# ------------------------ Helpers ------------------------
def qc_to_qasm(qc: QuantumCircuit) -> str:
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
    return qasm_text

# High-quality circuit image via matplotlib
_def_style = dict(border_radius="12px", shadow="0 6px 16px rgba(0,0,0,.06)")

def _circuit_image_html(qc: QuantumCircuit) -> str:
    fig = qc.draw(output='mpl')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=220)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"<img src='data:image/png;base64,{b64}' style='width:100%;max-width:1200px;border-radius:12px;box-shadow:0 6px 16px rgba(0,0,0,.06);' alt='Circuit Diagram'/>"

# ------------------------ Routes ------------------------
@app.route("/style.css")
def style_css():
    try:
        from flask import make_response
        with open('style.css', 'r', encoding='utf-8') as f:
            resp = make_response(f.read())
            resp.headers['Content-Type'] = 'text/css'
            return resp
    except Exception:
        return "", 404

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    qc = _get_qc()

    # Defaults for UI controls
    num_qubits = qc.num_qubits
    num_clbits = qc.num_clbits
    target_qubit = 0
    control_qubit = 1 if qc.num_qubits > 1 else 0
    rot_angle_pi = 0.5

    if request.method == "POST":
        action = request.form.get("action")
        try:
            num_qubits = int(request.form.get("num_qubits", num_qubits))
            num_clbits = int(request.form.get("num_clbits", num_clbits))
            target_qubit = int(request.form.get("target_qubit", target_qubit))
            control_qubit = int(request.form.get("control_qubit", control_qubit))
            rot_angle_pi = float(request.form.get("rot_angle_pi", rot_angle_pi))
        except Exception:
            pass

        # Clamp indices into valid range
        if num_qubits > 0:
            target_qubit = max(0, min(target_qubit, num_qubits - 1))
            control_qubit = max(0, min(control_qubit, num_qubits - 1))
            # If they collide and we have >=2 qubits, auto-pick a different control
            if num_qubits > 1 and control_qubit == target_qubit:
                control_qubit = (target_qubit + 1) % num_qubits

        if action == "resize":
            qc = QuantumCircuit(num_qubits, num_clbits)
            _set_qc(qc)
            message = "Circuit resized."
        elif action == "upload_qasm" and 'qasm_file' in request.files:
            file = request.files['qasm_file']
            try:
                qasm_content = file.read().decode('utf-8')
                qc = parse_qasm_string(qasm_content)
                _set_qc(qc)
                message = "QASM loaded."
            except Exception as e:
                message = f"Error: {str(e)}"
        elif action == "reset":
            qc = QuantumCircuit(num_qubits, num_clbits)
            _set_qc(qc)
            message = "Circuit reset."
        elif action == "clear":
            qc = QuantumCircuit(2, 2)
            _set_qc(qc)
            message = "Circuit cleared."
        elif action == "random":
            import random
            qc = QuantumCircuit(num_qubits, num_clbits)
            for _ in range(random.randint(3, 8)):
                gate = random.choice(['h', 'x', 'y', 'z', 'rx', 'ry', 'rz'])
                q = random.randint(0, num_qubits-1)
                if gate in ['rx', 'ry', 'rz']:
                    angle = random.uniform(0, 2*np.pi)
                    getattr(qc, gate)(angle, q)
                else:
                    getattr(qc, gate)(q)
            if num_qubits > 1:
                for _ in range(random.randint(1, 3)):
                    c = random.randint(0, num_qubits-1)
                    t = random.randint(0, num_qubits-1)
                    if c != t:
                        qc.cx(c, t)
            _set_qc(qc)
            message = "Random circuit generated."
        elif action in {"h","x","y","z","s","t","sdg","rx","ry","rz","cx","cy","cz"}:
            try:
                if action in {"h","x","y","z","s","t","sdg"}:
                    getattr(qc, action)(target_qubit)
                elif action in {"rx","ry","rz"}:
                    getattr(qc, action)(rot_angle_pi*np.pi, target_qubit)
                elif action in {"cx","cy","cz"} and num_qubits > 1 and control_qubit != target_qubit:
                    getattr(qc, action)(control_qubit, target_qubit)
                _set_qc(qc)
                message = f"Applied {action.upper()}"
            except Exception as e:
                message = f"Error applying {action}: {str(e)}"
        elif action == "preset_bell":
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            _set_qc(qc)
            message = "Bell state preset applied."
        elif action == "preset_ghz":
            qc = QuantumCircuit(3, 3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(1, 2)
            _set_qc(qc)
            message = "GHZ preset applied."

    # Circuit as text (ASCII) for clarity
    circuit_text = qc.draw(output='text')


    # Probability heatmap and state analysis
    rho = simulate_circuit(qc)
    n_states = 2**qc.num_qubits
    state_labels = [f"|{format(i, f'0{qc.num_qubits}b')}⟩" for i in range(n_states)]
    prob_df = pd.DataFrame(np.abs(rho.data) ** 2, index=state_labels, columns=state_labels)
    heatmap = px.imshow(prob_df, title="Probability Matrix |⟨i|ρ|j⟩|²", color_continuous_scale="Blues", aspect="auto", text_auto=True)
    heatmap.update_layout(height=500)
    heatmap_html = heatmap.to_html(full_html=False, include_plotlyjs='cdn')

    # Real and Imaginary heatmaps
    real_df = pd.DataFrame(np.real(rho.data), index=state_labels, columns=state_labels)
    real_map = px.imshow(real_df, title="Real Part Re(⟨i|ρ|j⟩)", color_continuous_scale="RdBu_r", aspect="auto", text_auto=True)
    real_map.update_layout(height=450)
    real_html = real_map.to_html(full_html=False, include_plotlyjs='cdn')

    imag_df = pd.DataFrame(np.imag(rho.data), index=state_labels, columns=state_labels)
    imag_map = px.imshow(imag_df, title="Imaginary Part Im(⟨i|ρ|j⟩)", color_continuous_scale="RdBu_r", aspect="auto", text_auto=True)
    imag_map.update_layout(height=450)
    imag_html = imag_map.to_html(full_html=False, include_plotlyjs='cdn')

    state_vector = rho.data.diagonal() if rho.num_qubits == 1 else rho.data.flatten()
    probs = np.abs(state_vector) ** 2
    prob_str = ", ".join([f"{p:.6f}" for p in probs])

    # Bloch spheres (Plotly 3D, same as large)
    bloch_html_blocks = []
    for i in range(qc.num_qubits):
        try:
            dm = get_single_qubit_dm(rho, i, qc.num_qubits)
            bloch = dm_to_bloch(dm)
            fig_b = bloch_plot(bloch, title=f"Qubit {i} Bloch Sphere")
            fig_b.update_layout(height=260, width=260, margin=dict(l=0, r=0, t=30, b=0))
            bloch_html_blocks.append(fig_b.to_html(full_html=False, include_plotlyjs=False))
        except Exception:
            continue

    # Purity and interpretation
    purity = float(np.real_if_close(np.trace(rho.data @ rho.data)))
    interpretation = ""
    non_zero_states = [(i, v, p) for i, (v, p) in enumerate(zip(state_vector, probs)) if abs(v) > 1e-10]
    if len(non_zero_states) == 1:
        b = format(non_zero_states[0][0], f'0{qc.num_qubits}b')
        interpretation = f"Computational basis state |{b}⟩"
    elif len(non_zero_states) == 2 and abs(abs(non_zero_states[0][1]) - abs(non_zero_states[1][1])) < 1e-10:
        interpretation = "Maximally entangled-like superposition"
    elif len(non_zero_states) == 4 and all(abs(abs(v) - 0.5) < 1e-10 for _, v, _ in non_zero_states):
        interpretation = "Uniform superposition of four basis states"
    else:
        interpretation = f"Superposition of {len(non_zero_states)} states"

    return render_template(
        "index.html",
        message=message,
        qc=qc,
        circuit_text=str(circuit_text),
        qasm_text=qc_to_qasm(qc),
        heatmap_html=heatmap_html,
        real_html=real_html,
        imag_html=imag_html,
        prob_str=prob_str,
        purity=purity,
        interpretation=interpretation,
        bloch_html_blocks=bloch_html_blocks,
        num_qubits=num_qubits,
        num_clbits=num_clbits,
        target_qubit=target_qubit,
        control_qubit=control_qubit,
        rot_angle_pi=rot_angle_pi,
    )

@app.route("/export_qasm")
def export_qasm():
    qasm_text = qc_to_qasm(_get_qc())
    from flask import make_response
    resp = make_response(qasm_text)
    resp.headers['Content-Type'] = 'text/plain'
    resp.headers['Content-Disposition'] = 'attachment; filename="quantum_circuit.qasm"'
    return resp



@app.route("/download_pdf")
def download_pdf():
    # Use the current UI circuit for PDF
    pdf_bytes = generate_pdf_report(_get_qc())
    return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf', as_attachment=True, download_name='quantum_report.pdf')

# Serve background image from project root
@app.route('/bg.jpg')
def serve_bg():
    return send_file('bg.jpg', mimetype='image/jpeg')


# Register Bloch API blueprint
from bloch_api import bloch_api
app.register_blueprint(bloch_api)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
