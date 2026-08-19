"""
Shared presentation helpers for the FUM dashboard.

Keeps Streamlit chrome, typography, and Plotly styling in one place so
the Cloud entry point and the local dashboard stay visually aligned.
"""

from __future__ import annotations

import html
import os
from typing import Optional, Sequence

import streamlit as st

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

INK = "#141311"
MUTED = "#767167"
LINE = "#E3DDD1"
BRAND = "#214C3A"
DANGER = "#9A2F2A"
WARN = "#8A5814"
LOW = "#2F6A4D"
MID = "#B45309"
CHARCOAL = "#161513"


def inject_theme() -> None:
    css_path = os.path.join(REPO_ROOT, "assets", "theme.css")
    with open(css_path, encoding="utf-8") as fh:
        st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)

    import streamlit.components.v1 as components

    # Keep this iframe 1px tall so Chrome still executes the script.
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;

          // Clean up stale sessionStorage key from a prior version that
          // caused the sidebar to start hidden on every page load.
          try { sessionStorage.removeItem("fum-sidebar-minimized"); } catch(e) {}

          // Use a data attribute on <body> to persist minimize state across
          // Streamlit reruns (the body element survives reruns). Unlike
          // sessionStorage this resets on page refresh, so the sidebar
          // always starts visible on a fresh load.
          let minimized = doc.body.dataset.fumSidebarMin === "1";

          // Grace period: let Streamlit finish its own initialisation
          // before we start intercepting collapse events. Without this,
          // Streamlit's startup can briefly set aria-expanded="false"
          // and our code would interpret it as a user-initiated collapse.
          const scriptStart = Date.now();
          const GRACE_MS = 1000;
          function pastGrace() { return Date.now() - scriptStart > GRACE_MS; }

          function sidebarEl() {
            return doc.querySelector('[data-testid="stSidebar"]');
          }

          // Force the sidebar to be "expanded" from Streamlit's perspective.
          function forceExpand() {
            const sb = sidebarEl();
            if (!sb) return;
            sb.setAttribute("aria-expanded", "true");
            // Clear every inline style Streamlit may have applied
            ["width","min-width","max-width","transform",
             "visibility","display"].forEach(function(p) {
              sb.style.removeProperty(p);
            });
          }

          // Apply our visual minimize/maximize state via a body class.
          function applyState() {
            doc.body.classList.toggle("fum-sidebar-minimized", minimized);
            doc.body.dataset.fumSidebarMin = minimized ? "1" : "0";
            const btn = ensureBtn();
            btn.classList.toggle("is-visible", minimized);
            btn.setAttribute("aria-label",
              minimized ? "Open navigation" : "Close navigation");
            btn.setAttribute("aria-hidden", minimized ? "false" : "true");
          }

          // After the grace period, attach a MutationObserver so that
          // clicking Streamlit's native X/close button is caught
          // instantly (instead of waiting up to 200ms for the next tick).
          let obsReady = false;
          function maybeSetupObserver() {
            if (obsReady || !pastGrace()) return;
            const sb = sidebarEl();
            if (!sb) return;
            // Disconnect any observer left by a previous Streamlit rerun
            if (sb._fumObs) { try { sb._fumObs.disconnect(); } catch(e) {} }
            const obs = new MutationObserver(function (mutations) {
              for (const m of mutations) {
                if (m.attributeName === "aria-expanded" &&
                    m.target.getAttribute("aria-expanded") === "false") {
                  forceExpand();
                  minimized = true;
                  applyState();
                }
              }
            });
            obs.observe(sb, { attributes: true, attributeFilter: ["aria-expanded"] });
            sb._fumObs = obs;
            obsReady = true;
          }

          // Ensure exactly one toggle button exists as a direct child of
          // document.body. Using body.querySelector(":scope > …") avoids
          // picking up duplicates that Streamlit creates in the content area.
          function ensureBtn() {
            var btn = doc.body.querySelector(":scope > .fum-nav-toggle");
            if (btn) return btn;
            // Remove any stale duplicates (e.g. from _html or prior reruns)
            doc.querySelectorAll(".fum-nav-toggle").forEach(function(el) { el.remove(); });
            btn = doc.createElement("button");
            btn.id = "fum-nav-toggle";
            btn.className = "fum-nav-toggle";
            btn.type = "button";
            btn.setAttribute("aria-label", "Open navigation");
            btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true"><path d="M3 5h12M3 9h12M3 13h12" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>';
            doc.body.appendChild(btn);
            btn.addEventListener("click", function (e) {
              e.preventDefault();
              e.stopPropagation();
              minimized = !minimized;
              if (!minimized) forceExpand();
              applyState();
            });
            return btn;
          }

          function tick() {
            maybeSetupObserver();
            // Safety net (only after grace period): if Streamlit collapsed
            // the sidebar between observer events, reverse it.
            if (pastGrace()) {
              var sb = sidebarEl();
              if (sb && sb.getAttribute("aria-expanded") === "false") {
                forceExpand();
                minimized = true;
              }
            }
            applyState();
          }

          setInterval(tick, 200);
          tick();
        })();
        </script>
        """,
        height=1,
        width=1,
    )


def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def _html(markup: str) -> None:
    # Streamlit's markdown parser inserts <p> tags around multiline HTML
    # and will collapse a CSS grid. Emit a single-line fragment instead.
    st.markdown(
        " ".join(line.strip() for line in markup.splitlines() if line.strip()),
        unsafe_allow_html=True,
    )


def configure_page(title: str = "FUM Risk Model") -> None:
    favicon = os.path.join(REPO_ROOT, "assets", "favicon.png")
    st.set_page_config(
        page_title=title,
        page_icon=favicon if os.path.exists(favicon) else None,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": "https://github.com/sichenz/dmh-fum-risk-model",
            "Report a bug": "https://github.com/sichenz/dmh-fum-risk-model/issues",
            "About": "Portfolio prototype for LA County behavioral health analytics. Synthetic data only.",
        },
    )


def brand_mark() -> str:
    return """
    <div class="brand">
      <div class="brand-mark" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M2.5 12.2L6.2 7.8L9.4 10.3L15.5 3.8" stroke="#FFFDF8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="15.5" cy="3.8" r="1.15" fill="#FFFDF8"/>
        </svg>
      </div>
      <div>
        <div class="brand-name">FUM Risk</div>
        <div class="brand-sub">LA County DMH</div>
      </div>
    </div>
    """


def render_brand() -> None:
    _html(brand_mark())


def hero(
    kicker: str,
    title: str,
    lede: str,
    chips: Sequence[str],
) -> None:
    chip_html = "".join(
        f'<span class="chip"><span class="swatch"></span>{_e(chip)}</span>'
        for chip in chips
    )
    _html(
        f"""
        <div class="hero">
          <div class="hero-copy">
            <div class="hero-kicker"><span class="dot"></span>{_e(kicker)}</div>
            <h1>{_e(title)}</h1>
            <p class="lede">{_e(lede)}</p>
            <div class="hero-meta">{chip_html}</div>
          </div>
          <div class="hero-product" aria-hidden="true">
            <div class="product-window">
              <div class="product-bar">
                <span class="traffic"><i></i><i></i><i></i></span>
                <span>Outreach queue</span>
                <span class="product-live">Live illustration</span>
              </div>
              <div class="product-row">
                <div>
                  <div class="pid">Token 8F2A · SPA 6</div>
                  <div class="pmeta">No appointment scheduled · 4 prior ED visits</div>
                </div>
                <span class="tag tag-critical">0.86</span>
              </div>
              <div class="product-row">
                <div>
                  <div class="pid">Token 11C0 · SPA 1</div>
                  <div class="pmeta">18-day wait · unstable housing</div>
                </div>
                <span class="tag tag-high">0.71</span>
              </div>
              <div class="product-row">
                <div>
                  <div class="pid">Token 90B3 · SPA 4</div>
                  <div class="pmeta">Limited English proficiency · no Rx at discharge</div>
                </div>
                <span class="tag tag-mid">0.58</span>
              </div>
              <div class="product-foot">3 of 184 flagged · synthetic cohort</div>
            </div>
          </div>
        </div>
        """
    )


def page_header(eyebrow: str, title: str, lede: str) -> None:
    _html(
        f"""
        <div class="page-head">
          <span class="eyebrow">{_e(eyebrow)}</span>
          <h2>{_e(title)}</h2>
          <p class="lede">{_e(lede)}</p>
        </div>
        """
    )


def metric_grid(items: Sequence[dict]) -> None:
    cells = []
    for item in items:
        hint = item.get("hint")
        hint_html = f'<div class="metric-hint">{_e(hint)}</div>' if hint else ""
        cells.append(
            f"""
            <div class="metric">
              <div class="metric-label">{_e(item["label"])}</div>
              <div class="metric-value">{_e(item["value"])}</div>
              {hint_html}
            </div>
            """
        )
    _html(f'<div class="metric-grid">{"".join(cells)}</div>')


def banner(text: str, kind: str = "info") -> None:
    _html(f'<div class="banner {kind}">{text}</div>')


def callout(text: str) -> None:
    _html(f'<div class="callout">{text}</div>')


def steps(items: Sequence[tuple[str, str, str]]) -> None:
    cards = []
    for num, title, body in items:
        cards.append(
            f"""
            <div class="step">
              <div class="step-num">{_e(num)}</div>
              <h3>{_e(title)}</h3>
              <p>{_e(body)}</p>
            </div>
            """
        )
    _html(f'<div class="steps">{"".join(cards)}</div>')


def sidebar_note(kicker: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="sidebar-note">
          <div class="k">{_e(kicker)}</div>
          <p>{_e(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def site_footer() -> None:
    _html(
        """
        <div class="site-footer">
          <div>
            <strong>FUM Risk Model</strong>
            Portfolio prototype for LA County behavioral health analytics.
          </div>
          <div>
            Synthetic cohort only. Not for clinical use without IRB approval,
            Privacy Officer sign-off, and LA County IT security review.
          </div>
        </div>
        """
    )


def risk_tier_label(score: float) -> str:
    if score >= 0.70:
        return "Critical"
    if score >= 0.55:
        return "High"
    return "Moderate"


def style_fig(fig, height: int = 300, show_legend: bool = False, title: Optional[str] = None):
    fig.update_layout(
        height=height,
        title=dict(text=title or "", font=dict(size=13, color=INK), x=0, xanchor="left")
        if title
        else None,
        margin=dict(l=8, r=8, t=36 if title else 12, b=8),
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color=INK, size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        xaxis=dict(gridcolor="#EDE8DE", zeroline=False, linecolor=LINE, ticks=""),
        yaxis=dict(gridcolor="#EDE8DE", zeroline=False, linecolor=LINE, ticks=""),
        bargap=0.18,
        hoverlabel=dict(bgcolor=CHARCOAL, font_size=12, font_family="Plus Jakarta Sans"),
        colorway=[CHARCOAL, BRAND, WARN, "#2C4664", DANGER],
    )
    return fig


def plot(fig, height: int = 300, show_legend: bool = False, title: Optional[str] = None) -> None:
    st.plotly_chart(
        style_fig(fig, height=height, show_legend=show_legend, title=title),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


def gap_delta(value: float, limit: float = 0.10) -> tuple[str, str]:
    if value < limit:
        return f"Within {limit:.2f}", "normal"
    return f"Above {limit:.2f}", "inverse"