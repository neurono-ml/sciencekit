// sciencekit — theme interactions (sidebar brand, scroll reveal, counters)

// Keep only Light and Dark in the theme picker (mdBook 0.5 renders the list
// server-side; extra options are removed here and "Coal" is relabeled).
(function () {
  function cleanThemeList() {
    var list = document.getElementById('mdbook-theme-list');
    if (!list || list.dataset.skCleaned) return;
    list.dataset.skCleaned = '1';
    ['default_theme', 'rust', 'navy', 'ayu'].forEach(function (suffix) {
      var button = document.getElementById('mdbook-theme-' + suffix);
      if (button && button.parentElement) button.parentElement.remove();
    });
    var coal = document.getElementById('mdbook-theme-coal');
    if (coal) coal.textContent = 'Dark';
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', cleanThemeList);
  } else {
    cleanThemeList();
  }
})();

(function () {
  var scrollbox = document.querySelector('.sidebar-scrollbox');
  if (scrollbox && !scrollbox.querySelector('.brand')) {
    var brand = document.createElement('div');
    brand.className = 'brand';
    brand.innerHTML =
      '<span class="brand-mark">sk</span>' +
      '<span><span class="brand-name">sciencekit</span>' +
      '<span class="brand-tagline">scikit-learn in Rust</span></span>';
    scrollbox.insertBefore(brand, scrollbox.firstChild);
  }
})();

document.addEventListener('DOMContentLoaded', function () {
  var reveals = document.querySelectorAll('.reveal');

  function animateCount(el) {
    var target = parseFloat(el.getAttribute('data-target')) || 0;
    var suffix = el.getAttribute('data-suffix') || '';
    var duration = 950;
    var start = null;
    function step(ts) {
      if (!start) start = ts;
      var progress = Math.min((ts - start) / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(target * eased) + suffix;
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
            entry.target.querySelectorAll('.js-count').forEach(animateCount);
          }
        });
      },
      { threshold: 0.12 }
    );
    reveals.forEach(function (el) { observer.observe(el); });
  } else {
    reveals.forEach(function (el) { el.classList.add('is-visible'); });
    document.querySelectorAll('.js-count').forEach(animateCount);
  }
});
