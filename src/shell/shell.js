/*
  «Три невозможные вещи» — shell.
  Mounts/unmounts scenes from window.SCENES, drives the overlay chrome,
  keyboard shortcuts, fullscreen, hidden-UI mode and the #hash deep link.
  Dependency-free, defensive: a scene that throws never takes the page down.
*/
(function () {
  'use strict';

  var SCENE_ORDER = ['gargantua', 'lenia', 'mandelbulb'];
  var ROMAN = { gargantua: 'I', lenia: 'II', mandelbulb: 'III' };
  var DEFAULT_SCENE = 'gargantua';
  var DEFAULT_ACCENT = '#f2b880';
  var VEIL_MS = 350;
  var FADE_MS = 180;

  var reduceMotion = false;
  var transitioning = false;
  var uiHidden = false;
  var current = null; // { id, def, handle }

  // DOM refs, filled in on boot.
  var stage, veil, chrome_, sceneUi;
  var btnHide, btnFullscreen, btnShow;
  var infoPanel, infoToggle, infoDetails;
  var sceneIndexEl, sceneNameEl, sceneTaglineEl, sceneAboutEl, sceneSpecEl, sceneHintEl;
  var switchButtons;

  function $(id) {
    return document.getElementById(id);
  }

  function availableScenes() {
    var s = window.SCENES || {};
    return SCENE_ORDER.filter(function (id) {
      return !!s[id];
    });
  }

  function isFormTarget(el) {
    if (!el) return false;
    var tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
  }

  function idFromHash() {
    var h = (location.hash || '').replace('#', '');
    return SCENE_ORDER.indexOf(h) !== -1 ? h : null;
  }

  function renderSpec(pairs) {
    sceneSpecEl.innerHTML = '';
    (pairs || []).forEach(function (pair) {
      var dt = document.createElement('dt');
      dt.textContent = pair[0];
      var dd = document.createElement('dd');
      dd.textContent = pair[1];
      sceneSpecEl.appendChild(dt);
      sceneSpecEl.appendChild(dd);
    });
  }

  function updateSwitcherState(id) {
    switchButtons.forEach(function (btn) {
      var isCurrent = btn.getAttribute('data-scene') === id;
      btn.setAttribute('aria-current', isCurrent ? 'true' : 'false');
    });
  }

  function fillInfoPanel(def) {
    sceneIndexEl.textContent = ROMAN[def.id] || '';
    sceneNameEl.textContent = def.name || '';
    sceneTaglineEl.textContent = def.tagline || '';
    sceneAboutEl.textContent = def.about || '';
    renderSpec(def.spec);
    sceneHintEl.textContent = def.hint || '';
  }

  function updateInfoPanel(def) {
    if (reduceMotion) {
      fillInfoPanel(def);
      return;
    }
    infoPanel.classList.add('is-fading');
    window.setTimeout(function () {
      fillInfoPanel(def);
      infoPanel.classList.remove('is-fading');
    }, FADE_MS);
  }

  function showFallback(message) {
    stage.innerHTML = '';
    var fb = document.createElement('div');
    fb.className = 'scene-fallback';
    fb.textContent = message;
    stage.appendChild(fb);
  }

  // Unmounts the current scene (if any), mounts `id`, updates chrome. No veil.
  function performSwitch(id) {
    var def = (window.SCENES || {})[id];
    if (!def) return;

    if (current && current.handle) {
      try {
        current.handle.unmount();
      } catch (err) {
        console.error('[shell] unmount error for ' + current.id, err);
      }
    }

    stage.innerHTML = '';
    sceneUi.innerHTML = '';

    var handle = null;
    try {
      handle = def.mount(stage, sceneUi) || null;
    } catch (err) {
      console.error('[shell] mount error for ' + id, err);
      showFallback('Сцена «' + (def.name || id) + '» сейчас не может запуститься. Попробуйте обновить страницу.');
      handle = null;
    }

    current = { id: id, def: def, handle: handle };
    updateInfoPanel(def);
    updateSwitcherState(id);
    document.documentElement.style.setProperty('--accent', def.accent || DEFAULT_ACCENT);
    try {
      history.replaceState(null, '', '#' + id);
    } catch (err) {
      /* ignore — e.g. sandboxed file:// edge cases */
    }
  }

  function switchTo(id) {
    if (transitioning) return;
    if (!window.SCENES || !window.SCENES[id]) return;
    if (current && current.id === id) return;

    if (reduceMotion) {
      performSwitch(id);
      return;
    }

    transitioning = true;
    veil.classList.add('is-visible');
    window.setTimeout(function () {
      performSwitch(id);
      // Let the fresh DOM paint once before fading the veil back out.
      requestAnimationFrame(function () {
        veil.classList.remove('is-visible');
        window.setTimeout(function () {
          transitioning = false;
        }, VEIL_MS);
      });
    }, VEIL_MS);
  }

  function stepScene(delta) {
    var scenes = availableScenes();
    if (!scenes.length) return;
    var idx = current ? scenes.indexOf(current.id) : -1;
    if (idx === -1) idx = 0;
    var next = (idx + delta + scenes.length) % scenes.length;
    switchTo(scenes[next]);
  }

  // ---------- hidden-UI mode ----------

  function setUiHidden(hidden) {
    uiHidden = hidden;
    chrome_.classList.toggle('is-hidden', hidden);
    chrome_.setAttribute('aria-hidden', hidden ? 'true' : 'false');
    btnHide.setAttribute('aria-pressed', hidden ? 'true' : 'false');
    btnShow.hidden = !hidden;
  }

  function toggleUiHidden() {
    setUiHidden(!uiHidden);
  }

  // ---------- fullscreen ----------

  function fullscreenSupported() {
    return !!(document.documentElement.requestFullscreen && document.exitFullscreen);
  }

  function toggleFullscreen() {
    if (!fullscreenSupported()) return;
    if (!document.fullscreenElement) {
      var p = document.documentElement.requestFullscreen();
      if (p && p.catch) p.catch(function () {});
    } else {
      var p2 = document.exitFullscreen();
      if (p2 && p2.catch) p2.catch(function () {});
    }
  }

  function onFullscreenChange() {
    btnFullscreen.setAttribute('aria-pressed', document.fullscreenElement ? 'true' : 'false');
  }

  // ---------- mobile info toggle ----------

  function setInfoExpanded(expanded) {
    infoToggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    infoToggle.textContent = expanded ? 'Свернуть' : 'Подробнее';
    infoPanel.classList.toggle('is-expanded', expanded);
  }

  function toggleInfoExpanded() {
    setInfoExpanded(infoToggle.getAttribute('aria-expanded') !== 'true');
  }

  // ---------- keyboard ----------

  function onKeydown(e) {
    if (e.defaultPrevented) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (isFormTarget(e.target)) return;

    switch (e.code) {
      case 'Digit1':
        switchTo(SCENE_ORDER[0]);
        break;
      case 'Digit2':
        switchTo(SCENE_ORDER[1]);
        break;
      case 'Digit3':
        switchTo(SCENE_ORDER[2]);
        break;
      case 'ArrowLeft':
        stepScene(-1);
        break;
      case 'ArrowRight':
        stepScene(1);
        break;
      case 'KeyH':
        toggleUiHidden();
        break;
      case 'KeyF':
        toggleFullscreen();
        break;
      case 'Escape':
        if (uiHidden) {
          setUiHidden(false);
        } else if (infoToggle.getAttribute('aria-expanded') === 'true') {
          setInfoExpanded(false);
        }
        break;
      default:
        return;
    }
  }

  // ---------- boot ----------

  function cacheDom() {
    stage = $('stage');
    veil = $('veil');
    chrome_ = $('chrome');
    sceneUi = $('scene-ui');
    btnHide = $('btn-hide');
    btnFullscreen = $('btn-fullscreen');
    btnShow = $('btn-show');
    infoPanel = $('info-panel');
    infoToggle = $('info-toggle');
    infoDetails = $('info-details');
    sceneIndexEl = $('scene-index');
    sceneNameEl = $('scene-name');
    sceneTaglineEl = $('scene-tagline');
    sceneAboutEl = $('scene-about');
    sceneSpecEl = $('scene-spec');
    sceneHintEl = $('scene-hint');
    switchButtons = Array.prototype.slice.call(document.querySelectorAll('.switch-btn'));
  }

  function wireEvents() {
    btnHide.addEventListener('click', toggleUiHidden);
    btnShow.addEventListener('click', toggleUiHidden);

    if (fullscreenSupported()) {
      btnFullscreen.addEventListener('click', toggleFullscreen);
      document.addEventListener('fullscreenchange', onFullscreenChange);
    } else {
      btnFullscreen.hidden = true;
    }

    infoToggle.addEventListener('click', toggleInfoExpanded);

    switchButtons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        switchTo(btn.getAttribute('data-scene'));
      });
    });

    document.addEventListener('keydown', onKeydown);
  }

  function boot() {
    cacheDom();
    wireEvents();

    try {
      reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch (err) {
      reduceMotion = false;
    }

    var scenes = availableScenes();

    switchButtons.forEach(function (btn) {
      var id = btn.getAttribute('data-scene');
      btn.hidden = scenes.indexOf(id) === -1;
    });

    if (!scenes.length) {
      showFallback('Сцены не найдены. Проверьте, что файлы src/scenes/*.js подключены.');
      return;
    }

    var initial = idFromHash();
    if (!initial || scenes.indexOf(initial) === -1) {
      initial = scenes.indexOf(DEFAULT_SCENE) !== -1 ? DEFAULT_SCENE : scenes[0];
    }

    performSwitch(initial);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
