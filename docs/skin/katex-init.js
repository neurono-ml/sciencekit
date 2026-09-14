// sciencekit — client-side KaTeX rendering for LaTeX equations
//
// mdBook does not render LaTeX natively. This script uses KaTeX's auto-render
// extension (katex-auto-render.min.js, loaded via book.toml) to render inline
// `$...$` and display `$$...$$` equations, then annotates every rendered
// equation with an accessible aria-label that reads the equation as spoken
// content (not the raw LaTeX) — e.g. for \int f(x)\,dx = g(x) the label reads
// "the integral of f of x d x equals g of x", so screen readers announce the
// math's meaning.

(function () {
  var DELIMITERS = [
    { left: '$$', right: '$$', display: true },
    { left: '$', right: '$', display: false },
    { left: '\\(', right: '\\)', display: false },
    { left: '\\[', right: '\\]', display: true },
  ];

  function renderMath() {
    if (typeof window.renderMathInElement === 'undefined') {
      setTimeout(renderMath, 100);
      return;
    }
    try {
      window.renderMathInElement(document.body, {
        delimiters: DELIMITERS,
        throwOnError: false,
        strict: false,
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
      });
    } catch (err) {
      return;
    }
    annotate();
  }

  // Give every rendered equation an accessible, plain-language label.
  function annotate() {
    document.querySelectorAll('.katex').forEach(function (el) {
      if (el.dataset.skLabeled) return;
      el.dataset.skLabeled = '1';
      var ann = el.querySelector('annotation[encoding="application/x-tex"]');
      var tex = ann ? ann.textContent : '';
      el.setAttribute('role', 'img');
      el.setAttribute('aria-label', speak(tex));
    });
  }

  // A pragmatic plain-language reading of LaTeX for screen readers.
  function speak(tex) {
    var s = tex
      .replace(/\\int/g, 'the integral of ')
      .replace(/\\sum/g, 'the sum of ')
      .replace(/\\prod/g, 'the product of ')
      .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, ' the fraction of $1 over $2 ')
      .replace(/\\sqrt\{([^{}]*)\}/g, 'the square root of $1')
      .replace(/\\left|\\right/g, '')
      .replace(/\\,|\\;|\\!|\\quad|\\qquad/g, ' ')
      .replace(/\\cdot/g, ' times ')
      .replace(/\\times/g, ' times ')
      .replace(/\\approx/g, ' approximately equal to ')
      .replace(/\\geq?/g, ' greater than or equal to ')
      .replace(/\\leq?/g, ' less than or equal to ')
      .replace(/\\neq/g, ' not equal to ')
      .replace(/\\in/g, ' in ')
      .replace(/\\alpha/g, ' alpha ').replace(/\\beta/g, ' beta ')
      .replace(/\\lambda/g, ' lambda ').replace(/\\mu/g, ' mu ')
      .replace(/\^2/g, ' squared ')
      .replace(/\^\{?2\}?/g, ' squared ')
      .replace(/\^\{?([^}]*)\}?/g, ' to the power of $1 ')
      .replace(/_\{([^}]*)\}/g, ' sub $1 ')
      .replace(/_([A-Za-z0-9])/g, ' sub $1 ')
      .replace(/[{}]/g, ' ')
      .replace(/\\/g, ' ')
      .replace(/[|]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    return s;
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderMath);
  } else {
    renderMath();
  }
})();