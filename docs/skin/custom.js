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

// sciencekit — font-size control (A- / A / A+).
// Bounded prose scaling persisted in localStorage under 'sk-font-scale'.
// Steps: 0 (small, default) → 1 → 2 (large); applied as
// html[data-sk-font-scale="n"] and consumed by custom.css.
(function () {
  var STORAGE_KEY = 'sk-font-scale';
  var MINIMUM_SCALE = 0;
  var MAXIMUM_SCALE = 2;
  var DEFAULT_SCALE = 0;

  function readScale() {
    try {
      var raw = window.localStorage.getItem(STORAGE_KEY);
      var parsed = parseInt(raw, 10);
      if (parsed >= MINIMUM_SCALE && parsed <= MAXIMUM_SCALE) return parsed;
    } catch (error) { /* storage unavailable — fall through */ }
    return DEFAULT_SCALE;
  }

  function applyScale(scale) {
    document.documentElement.setAttribute('data-sk-font-scale', String(scale));
    try { window.localStorage.setItem(STORAGE_KEY, String(scale)); }
    catch (error) { /* storage unavailable — attribute still applies */ }
    document.querySelectorAll('.sk-font-btn').forEach(function (button) {
      var action = button.getAttribute('data-sk-font-action');
      if (action === 'reset') {
        button.setAttribute('aria-pressed', scale === DEFAULT_SCALE ? 'true' : 'false');
      } else {
        button.disabled =
          (action === 'decrease' && scale <= MINIMUM_SCALE) ||
          (action === 'increase' && scale >= MAXIMUM_SCALE);
      }
    });
  }

  function injectControls() {
    var bar = document.querySelector('#mdbook-menu-bar .left-buttons');
    if (!bar || bar.querySelector('.sk-font-btn')) return;
    var group = document.createElement('div');
    group.className = 'sk-font-group';
    group.setAttribute('role', 'group');
    group.setAttribute('aria-label', 'Text size');
    group.innerHTML =
      '<button class="icon-button sk-font-btn" type="button" data-sk-font-action="decrease" title="Decrease text size" aria-label="Decrease text size">A−</button>' +
      '<button class="icon-button sk-font-btn" type="button" data-sk-font-action="reset" title="Reset text size" aria-label="Reset text size">A</button>' +
      '<button class="icon-button sk-font-btn" type="button" data-sk-font-action="increase" title="Increase text size" aria-label="Increase text size">A+</button>';
    bar.appendChild(group);
    group.addEventListener('click', function (event) {
      var button = event.target.closest('.sk-font-btn');
      if (!button) return;
      var action = button.getAttribute('data-sk-font-action');
      var scale = readScale();
      if (action === 'decrease') scale = Math.max(MINIMUM_SCALE, scale - 1);
      else if (action === 'increase') scale = Math.min(MAXIMUM_SCALE, scale + 1);
      else scale = DEFAULT_SCALE;
      applyScale(scale);
    });
    applyScale(readScale());
  }

  applyScale(readScale());
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectControls);
  } else {
    injectControls();
  }
})();

// sciencekit — sidebar accordion.
// mdBook renders every nested ol.section expanded. This collapses sections
// into a single-open accordion: expanding one closes its siblings, the
// section holding the active chapter opens on load, and state persists in
// localStorage. Without JavaScript all sections stay visible (no-JS fallback).
(function () {
  var STORAGE_KEY = 'sk-sidebar-open';
  var SECTION_SELECTOR = 'li.chapter-item > ol.section';

  function sectionKey(list) {
    var parent = list.parentElement;
    var link = parent && parent.querySelector(':scope > span > a, :scope > a');
    return (link && link.getAttribute('href')) || null;
  }

  function readOpenKeys() {
    try {
      var raw = window.localStorage.getItem(STORAGE_KEY);
      var parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed;
    } catch (error) { /* storage unavailable — fall through */ }
    return null;
  }

  function writeOpenKeys(keys) {
    try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(keys)); }
    catch (error) { /* ignore persistence failures */ }
  }

  function setOpen(list, open) {
    var item = list.parentElement;
    item.classList.toggle('expanded', open);
    item.classList.toggle('sk-collapsed', !open);
    var toggle = item.querySelector(':scope > .sk-section-toggle');
    if (toggle) toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  function collectOpenKeys(scope) {
    var keys = [];
    scope.querySelectorAll(SECTION_SELECTOR).forEach(function (list) {
      if (list.parentElement.classList.contains('expanded')) {
        var key = sectionKey(list);
        if (key) keys.push(key);
      }
    });
    return keys;
  }

  function enhance(scope) {
    var sections = scope.querySelectorAll(SECTION_SELECTOR);
    if (!sections.length || scope.dataset.skAccordion) return;
    scope.dataset.skAccordion = '1';
    var persisted = readOpenKeys();

    sections.forEach(function (list) {
      var item = list.parentElement;
      var link = item.querySelector(':scope > span > a, :scope > a');
      if (!link || item.querySelector(':scope > .sk-section-toggle')) return;
      var toggle = document.createElement('button');
      toggle.className = 'sk-section-toggle';
      toggle.type = 'button';
      toggle.setAttribute('aria-expanded', 'true');
      toggle.setAttribute('aria-label', 'Toggle section: ' + link.textContent.trim());
      item.insertBefore(toggle, item.firstChild);
      toggle.addEventListener('click', function () {
        toggleSection(item, list, scope);
      });
      // Clicking the section link of the page already being viewed toggles
      // the submenu instead of reloading the same page (standard docs
      // behavior). Links to other pages navigate normally; the freshly
      // loaded page then single-opens around the active chapter.
      link.addEventListener('click', function (event) {
        if (event.defaultPrevented || event.button !== 0 || event.metaKey ||
            event.ctrlKey || event.shiftKey || event.altKey) return;
        var target = link.getAttribute('href') || '';
        if (target.charAt(0) === '#') return;
        var current = window.location.pathname.split('/').pop() || 'index.html';
        var destination = target.split('/').pop().split('#')[0] || 'index.html';
        if (destination === current) {
          event.preventDefault();
          toggleSection(item, list, scope);
        }
      });
    });

    function toggleSection(item, list, scope) {
      var willOpen = !item.classList.contains('expanded');
      if (willOpen) {
        // Single-open: close sibling sections at the same level.
        var siblings = Array.from(item.parentElement.children).filter(function (child) {
          return child !== item && child.matches('li.chapter-item');
        });
        siblings.forEach(function (sibling) {
          var nested = sibling.querySelector(':scope > ol.section');
          if (nested) setOpen(nested, false);
        });
      }
      setOpen(list, willOpen);
      writeOpenKeys(collectOpenKeys(scope));
    }

    if (persisted) {
      sections.forEach(function (list) {
        setOpen(list, persisted.indexOf(sectionKey(list)) !== -1);
      });
    } else {
      // Default: only the section holding the active chapter stays open.
      var active = scope.querySelector('li.chapter-item.active, a.active');
      sections.forEach(function (list) {
        var holdsActive = active && list.contains(active);
        // Keep top-level sections open when nothing is active yet.
        setOpen(list, holdsActive || !active);
      });
    }
  }

  function boot() {
    var sidebar = document.getElementById('mdbook-sidebar');
    if (!sidebar) return;
    enhance(sidebar);
    // The sidebar is injected asynchronously (toc.js); observe until enhanced.
    if (!sidebar.dataset.skAccordion) {
      var observer = new MutationObserver(function () {
        enhance(sidebar);
        if (sidebar.dataset.skAccordion) observer.disconnect();
      });
      observer.observe(sidebar, { childList: true, subtree: true });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
