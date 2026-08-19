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

    # Always-on control in the main document. Streamlit's native reopen
    # button lives inside the header and Chrome clips it after deploy.
    _html(
        """
        <button type="button" class="fum-nav-toggle" id="fum-nav-toggle" aria-label="Open navigation">
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
            <path d="M3 5h12M3 9h12M3 13h12" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/>
          </svg>
        </button>
        """
    )

    import streamlit.components.v1 as components

    # Keep this iframe 1px tall so Chrome still executes the script.
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;

          function sidebarEl() {
            return doc.querySelector('[data-testid="stSidebar"]');
          }

          function isCollapsed() {
            const sb = sidebarEl();
            if (!sb) return false;
            return sb.getAttribute("aria-expanded") === "false";
          }

          // Try to find Streamlit's native expand/collapse button across
          // different Streamlit versions.
          function nativeToggle() {
            const selectors = [
              '[data-testid="stExpandSidebarButton"]',
              '[data-testid="stSidebarCollapsedControl"] button',
              '[data-testid="stSidebarCollapsedControl"]',
              '[data-testid="collapsedControl"] button',
              '[data-testid="collapsedControl"]',
              '[data-testid="stSidebarCollapseButton"] button',
              '[data-testid="stSidebarCollapseButton"]',
            ];
            for (const sel of selectors) {
              const el = doc.querySelector(sel);
              if (el) return el;
            }
            // Fallback: any button whose aria-label mentions "sidebar"
            const candidates = doc.querySelectorAll(
              'button[aria-label], [role="button"][aria-label]'
            );
            for (const el of candidates) {
              const label = (el.getAttribute("aria-label") || "").toLowerCase();
              if (label.indexOf("sidebar") !== -1) return el;
            }
            return null;
          }

          // Directly expand the sidebar by setting aria-expanded and
          // clearing any inline styles that Streamlit may have applied.
          function expandSidebar() {
            const sb = sidebarEl();
            if (!sb) return;
            sb.setAttribute("aria-expanded", "true");
            // Clear any inline transform/visibility that Streamlit sets
            sb.style.removeProperty("transform");
            sb.style.removeProperty("visibility");
            sb.style.removeProperty("width");
            sb.style.removeProperty("min-width");
            sb.style.removeProperty("max-width");
          }

          // Collapse the sidebar. Try native toggle first; if that
          // fails, set the attribute directly and let CSS handle it.
          function collapseSidebar() {
            const sb = sidebarEl();
            if (!sb) return;
            const native = nativeToggle();
            if (native) {
              native.click();
              // Double-check after a tick
              setTimeout(function () {
                if (!isCollapsed()) {
                  sb.setAttribute("aria-expanded", "false");
                }
              }, 100);
            } else {
              sb.setAttribute("aria-expanded", "false");
            }
          }

          // Ensure our toggle button exists in document.body (not
          // inside the sidebar or an iframe), so it survives collapse.
          function ensureBtn() {
            let btn = doc.getElementById("fum-nav-toggle");
            if (!btn) {
              btn = doc.createElement("button");
              btn.id = "fum-nav-toggle";
              btn.className = "fum-nav-toggle";
              btn.type = "button";
              btn.setAttribute("aria-label", "Open navigation");
              btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M3 5h12M3 9h12M3 13h12" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>';
            }
            if (btn.parentElement !== doc.body) doc.body.appendChild(btn);
            if (!btn.dataset.bound) {
              btn.dataset.bound = "1";
              btn.addEventListener("click", function (e) {
                e.preventDefault();
                e.stopPropagation();

                if (isCollapsed()) {
                  // Try native toggle first
                  const native = nativeToggle();
                  if (native) native.click();
                  // Guaranteed fallback after a short delay
                  setTimeout(function () {
                    if (isCollapsed()) {
                      expandSidebar();
                    }
                  }, 150);
                } else {
                  collapseSidebar();
                }
              });
            }
            return btn;
          }

          function tick() {
            const btn = ensureBtn();
            const collapsed = isCollapsed();
            doc.body.classList.toggle("fum-sidebar-collapsed", collapsed);
            btn.classList.toggle("is-visible", collapsed);
            btn.setAttribute("aria-label", collapsed ? "Open navigation" : "Close navigation");
            btn.setAttribute("aria-hidden", collapsed ? "false" : "true");
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