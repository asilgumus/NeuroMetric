/* NeuroMetric showcase site — visual interactions only (no analysis) */
(function () {
  "use strict";

  // Reveal animation only engages when JS runs
  document.documentElement.classList.add("js");

  var yearEl = document.getElementById("yil");
  if (yearEl) yearEl.textContent = String(new Date().getFullYear());

  // Top bar shadow
  var nav = document.querySelector(".nav");
  var onScroll = function () {
    if (nav) nav.classList.toggle("scrolled", window.scrollY > 4);
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // Mobile menu
  var burger = document.getElementById("burger");
  var mnav = document.getElementById("mnav");
  if (burger && mnav) {
    burger.addEventListener("click", function () {
      var open = burger.getAttribute("aria-expanded") === "true";
      burger.setAttribute("aria-expanded", String(!open));
      mnav.hidden = open;
      mnav.style.display = open ? "" : "grid";
    });
    mnav.addEventListener("click", function (e) {
      if (e.target.closest("a")) {
        burger.setAttribute("aria-expanded", "false");
        mnav.hidden = true;
        mnav.style.display = "";
      }
    });
  }

  // The selection syncs into the dashboard link — data lives statically in report.html; no real analysis
  var sel = document.getElementById("caseSel");
  var dashBtn = document.getElementById("dashBtn");
  if (sel && dashBtn) {
    var setHref = function () {
      dashBtn.href = "report.html?case=" + encodeURIComponent(sel.value);
    };
    sel.addEventListener("change", setHref);
    setHref();
  }

  // "Trusted" testimonial carousel — visual only, infinite loop
  var vp = document.getElementById("tstViewport");
  var track = document.getElementById("tstTrack");
  var dotsEl = document.getElementById("tstDots");
  if (vp && track && dotsEl) {
    var slides = Array.prototype.slice.call(track.children);
    var n = slides.length;
    var reduceMq = window.matchMedia("(prefers-reduced-motion: reduce)");

    // infinite loop: clone the last slide to the front, every slide to the end
    var firstClone = slides[n - 1].cloneNode(true);
    firstClone.setAttribute("aria-hidden", "true");
    firstClone.classList.remove("is-active");
    track.insertBefore(firstClone, track.firstChild);
    slides.forEach(function (sl) {
      var cl = sl.cloneNode(true);
      cl.setAttribute("aria-hidden", "true");
      cl.classList.remove("is-active");
      track.appendChild(cl);
    });

    var idx = 1;              // 1..n real slides
    var timer = null;
    var all = function () { return Array.prototype.slice.call(track.children); };

    var paint = function () {
      var t = track.children[idx];
      if (!t) return;
      all().forEach(function (s, k) { s.classList.toggle("is-active", k === idx); });
      Array.prototype.forEach.call(dotsEl.children, function (d, k) {
        d.classList.toggle("on", k === (idx - 1 + n) % n);
      });
      var d = t.offsetLeft - (vp.clientWidth - t.offsetWidth) / 2;
      track.style.transform = "translateX(" + -d + "px)";
    };
    var silent = function () {
      track.style.transition = "none";
      paint();
      void track.offsetWidth;
      track.style.transition = "";
    };
    var restart = function () {
      if (timer) clearInterval(timer);
      timer = null;
      if (reduceMq.matches) return;
      timer = setInterval(function () { go(idx + 1); }, 6800);
    };
    var go = function (k, fromUser) {
      idx = k;
      paint();
      if (fromUser) restart();
    };
    var wrapReset = function (expected, real) {
      window.setTimeout(function () {
        if (idx === expected) { idx = real; silent(); restart(); }
      }, 570);
    };
    var goWrap = function (dir) {
      var target = idx + dir;
      if (target > n) {             // past the last real slide; advance through clones
        go(target, true);
        wrapReset(target, 1);
      } else if (target < 1) {
        go(0, true);
        wrapReset(0, n);
      } else go(target, true);
    };

    slides.forEach(function (_, k) {
      var b = document.createElement("button");
      b.type = "button";
      b.setAttribute("aria-label", "Quote " + (k + 1));
      b.addEventListener("click", function () {
        if (idx >= n + 1) { idx = 1 + k; silent(); }
        else if (idx < 1) { idx = 1 + k; silent(); }
        else go(k + 1, true);
      });
      dotsEl.appendChild(b);
    });

    var prev = document.getElementById("tstPrev");
    var next = document.getElementById("tstNext");
    if (prev) prev.addEventListener("click", function () { goWrap(-1); });
    if (next) next.addEventListener("click", function () { goWrap(1); });

    window.addEventListener("resize", function () { silent(); });
    track.addEventListener("transitionend", function (e) {
      if (e.propertyName !== "transform") return;
      if (idx > n) { idx -= n; silent(); }
      else if (idx < 1) { idx += n; silent(); }
    });
    silent();
    restart();
  }

  // Scroll reveal
  var rvs = Array.prototype.slice.call(document.querySelectorAll(".rv"));
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          en.target.classList.add("in");
          io.unobserve(en.target);
        }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });
    rvs.forEach(function (el) { io.observe(el); });
  } else {
    rvs.forEach(function (el) { el.classList.add("in"); });
  }
})();