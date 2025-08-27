// Enable full screen for Bloch spheres


function enableBlochFullscreen() {
  document.querySelectorAll('.bloch-card').forEach(function(card) {
    card.style.cursor = 'zoom-in';
    if (card._fullscreenEnabled) return; // Prevent double binding
    card._fullscreenEnabled = true;
    card.addEventListener('click', function() {
      // Try Plotly full screen if available
      const plotlyDiv = card.querySelector('.js-plotly-plot');
      if (plotlyDiv && window.Plotly && Plotly.Plots && Plotly.Plots.fullscreen) {
        Plotly.Plots.fullscreen(plotlyDiv);
        return;
      }
      if (plotlyDiv) {
        // Fallback: open in new window/tab as static image
        Plotly.toImage(plotlyDiv, {format: 'png', width: 800, height: 800}).then(function(dataUrl) {
          const win = window.open();
          win.document.write('<img src="' + dataUrl + '" style="max-width:100vw;max-height:100vh;display:block;margin:auto;background:#fff;">');
        });
        return;
      }
      // Otherwise, clone the first child (e.g. img/svg)
      const plot = card.firstElementChild ? card.firstElementChild.cloneNode(true) : card.cloneNode(true);
      const overlay = document.createElement('div');
      overlay.style.position = 'fixed';
      overlay.style.top = 0;
      overlay.style.left = 0;
      overlay.style.width = '100vw';
      overlay.style.height = '100vh';
      overlay.style.background = 'rgba(0,0,0,0.85)';
      overlay.style.display = 'flex';
      overlay.style.alignItems = 'center';
      overlay.style.justifyContent = 'center';
      overlay.style.zIndex = 9999;
      overlay.style.cursor = 'zoom-out';
      plot.style.maxWidth = '90vw';
      plot.style.maxHeight = '90vh';
      plot.style.width = 'auto';
      plot.style.height = 'auto';
      plot.style.background = '#fff';
      plot.style.borderRadius = '12px';
      plot.style.boxShadow = '0 8px 32px rgba(0,0,0,0.25)';
      overlay.appendChild(plot);
      overlay.addEventListener('click', function() {
        overlay.remove();
      });
      document.body.appendChild(overlay);
    });
  });
}

window.addEventListener('DOMContentLoaded', function() {
  enableBlochFullscreen();
  // Observe for dynamic changes to .bloch-grid
  var grid = document.querySelector('.bloch-grid');
  if (grid) {
    var observer = new MutationObserver(function() {
      enableBlochFullscreen();
    });
    observer.observe(grid, { childList: true, subtree: true });
  }
});
