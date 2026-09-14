// sciencekit — client-side Mermaid rendering for mdBook
//
// mdBook does not render ```mermaid fenced blocks natively; it emits them as
// <pre><code class="language-mermaid">. This script loads the bundled
// mermaid.min.js (already on the page via book.toml additional-js) and renders
// every such block into an accessible <div class="mermaid"> as an SVG, with a
// role="img" and aria-label from the code's accTitle/accDescr (or a fallback).
//
// It also exposes the diagram to screen readers: the SVG carries an aria-label
// describing the *content*, not the raw mermaid source.

(function () {
  function renderAll() {
    if (typeof window.mermaid === 'undefined') {
      // mermaid.min.js not loaded yet — retry once after a tick.
      setTimeout(renderAll, 100);
      return;
    }
    var blocks = document.querySelectorAll('pre code.language-mermaid');
    if (!blocks.length) return;
    window.mermaid.initialize({
      startOnLoad: false,
      theme: 'dark',
      securityLevel: 'loose',
      fontFamily: 'inherit',
    });
    var index = 0;
    blocks.forEach(function (code) {
      var pre = code.parentElement;
      if (pre.dataset.skMermaid) return;
      pre.dataset.skMermaid = '1';
      var source = code.textContent;
      var id = 'sk-mermaid-' + index++;
      window.mermaid
        .render(id, source)
        .then(function (result) {
          var container = document.createElement('div');
          container.className = 'mermaid';
          container.innerHTML = result.svg;
          // Accessibility: describe the content as read, not the raw source.
          var svg = container.querySelector('svg');
          if (svg) {
            svg.setAttribute('role', 'img');
            var title = extractTitle(source);
            svg.setAttribute('aria-label', title);
          }
          pre.replaceWith(container);
        })
        .catch(function (err) {
          var box = document.createElement('div');
          box.className = 'sk-box sk-box--danger';
          box.textContent = 'Mermaid diagram failed to render: ' + err.message;
          pre.replaceWith(box);
        });
    });
  }

  function extractTitle(source) {
    // Prefer the mermaid accTitle/accDescr; else derive a short label.
    var m = source.match(/\baccTitle:\s*(.+)/);
    if (m) return m[1].trim();
    var d = source.match(/\baccDescr:\s*(.+)/);
    if (d) return d[1].trim();
    var first = source.split('\n').find(function (l) { return l.trim().length > 0; });
    return (first || 'Mermaid diagram').trim();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderAll);
  } else {
    renderAll();
  }
})();