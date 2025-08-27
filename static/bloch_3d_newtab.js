// Open 3D Bloch sphere in a new tab using Plotly JSON from /bloch3d

function enableBloch3DNewTab() {
  document.querySelectorAll('.bloch-3d-btn').forEach(function(btn) {
    if (btn._bloch3dEnabled) return;
    btn._bloch3dEnabled = true;
    btn.addEventListener('click', function(e) {
      e.preventDefault();
      const card = btn.closest('.bloch-card');
      const qubit = card ? card.getAttribute('data-qubit') : 0;
      window.open(`/bloch3d_html?qubit=${qubit}`, '_blank');
    });
  });
}
window.addEventListener('DOMContentLoaded', enableBloch3DNewTab);
