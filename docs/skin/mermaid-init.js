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
//
// Contrast policy (fixed palette): diagrams use the same fixed --sk-*
// fill+text pairs in both book themes, so a single contrast pass covers
// light and dark. The base theme below only sets *defaults* for nodes,
// edges, and edge labels that carry no explicit style; per-diagram
// classDef/style directives take precedence.

(function () {
  // Fixed defaults — dark text on light fills, neutral edges. Every pair
  // targets a contrast ratio >= 4.5:1 and is legible on both the light
  // (#ffffff) and dark (#0b0b0e) page backgrounds.
  var SK_THEME_VARIABLES = {
    fontFamily: 'inherit',
    fontSize: '13px',
    textColor: '#18181b',
    primaryTextColor: '#0c4a6e',
    primaryColor: '#e0f2fe',
    primaryBorderColor: '#0284c7',
    secondaryTextColor: '#14532d',
    secondaryColor: '#dcfce7',
    secondaryBorderColor: '#059669',
    tertiaryTextColor: '#3b0764',
    tertiaryColor: '#ede9fe',
    tertiaryBorderColor: '#7c3aed',
    lineColor: '#52525c',
    edgeLabelBackground: '#f4f4f5',
    clusterBkg: '#f4f4f5',
    clusterBorder: '#a8a8b3',
    nodeBorder: '#0284c7',
    mainBkg: '#e0f2fe',
  };

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
      theme: 'base',
      themeVariables: SK_THEME_VARIABLES,
      securityLevel: 'loose',
      fontFamily: 'inherit',
      flowchart: {
        htmlLabels: true,
        useMaxWidth: true,
        wrap: true,
        padding: 8,
      },
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
          container.setAttribute('role', 'figure');
          container.setAttribute('data-sk-source', source);
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

  function boot() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', renderWhenFontsReady);
    } else {
      renderWhenFontsReady();
    }
  }

  // Mermaid measures node text with the font available at render time. The
  // book's Inter webfont loads asynchronously; rendering before it arrives
  // makes every measurement too narrow and clips node text inside its
  // foreignObject viewport. Wait for the faces first (with a timeout
  // fallback so offline builds still render with fallback metrics).
  function renderWhenFontsReady() {
    var rendered = false;
    function done() {
      if (rendered) return;
      rendered = true;
      renderAll();
    }
    setTimeout(done, 1500);
    if (document.fonts && document.fonts.load) {
      try {
        Promise.all([
          document.fonts.load('400 13px Inter'),
          document.fonts.load('500 13px Inter'),
          document.fonts.load('600 13px Inter'),
          document.fonts.load('700 13px Inter'),
        ]).then(done, done);
      } catch (error) {
        done();
      }
    } else {
      done();
    }
  }

  // Re-render once all fonts settle (e.g. a face arriving after the gated
  // render). Containers stash their source in data-sk-source so the diagram
  // can be rebuilt; renderAll skips blocks already rendered.
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(function () {
      var containers = document.querySelectorAll('div.mermaid');
      if (!containers.length) return;
      var rebuilt = false;
      containers.forEach(function (container) {
        var source = container.getAttribute('data-sk-source');
        if (!source) return;
        var pre = document.createElement('pre');
        var code = document.createElement('code');
        code.className = 'language-mermaid';
        code.textContent = source;
        pre.appendChild(code);
        container.replaceWith(pre);
        rebuilt = true;
      });
      if (rebuilt) renderAll();
    });
  }

  boot();
})();
