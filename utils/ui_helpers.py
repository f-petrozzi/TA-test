"""UI helpers for theme colors, CSS injection, and auto-scrolling."""
from functools import lru_cache
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib

BASE_DIR = Path(__file__).resolve().parent.parent


def adjust_hex_color(color: str, factor: float) -> str:
    """Lighten (factor>0) or darken (factor<0) a hex color."""
    if not color:
        return color
    hex_value = color.strip().lstrip("#")
    if len(hex_value) not in (3, 6):
        return color
    if len(hex_value) == 3:
        hex_value = "".join(ch * 2 for ch in hex_value)
    comps = [int(hex_value[i : i + 2], 16) for i in (0, 2, 4)]
    out = []
    for comp in comps:
        if factor >= 0:
            comp = comp + (255 - comp) * min(factor, 1)
        else:
            comp = comp * max(1 + factor, 0)
        out.append(int(max(0, min(255, round(comp)))))
    return "#{:02x}{:02x}{:02x}".format(*out)


@lru_cache(maxsize=1)
def load_theme_colors() -> dict[str, str]:
    """Load theme colors from .streamlit/config.toml."""
    defaults = {
        "primary": "#006747",
        "background": "#f7f8f7",
        "secondary": "#e7efe5",
        "text": "#111111",
    }
    config_path = BASE_DIR / ".streamlit" / "config.toml"
    if config_path.exists():
        try:
            data = tomllib.loads(config_path.read_text(encoding="utf-8"))
            theme = data.get("theme", {})
            defaults["primary"] = theme.get("primaryColor", defaults["primary"])
            defaults["background"] = theme.get("backgroundColor", defaults["background"])
            defaults["secondary"] = theme.get("secondaryBackgroundColor", defaults["secondary"])
            defaults["text"] = theme.get("textColor", defaults["text"])
        except (tomllib.TOMLDecodeError, OSError):
            pass
    defaults["primary_dark"] = adjust_hex_color(defaults["primary"], -0.2)
    return defaults


def inject_global_styles() -> None:
    """Inject global CSS styles with theme variables."""
    css_path = BASE_DIR / "styles.css"
    chunks = []
    colors = load_theme_colors()
    chunks.append(
        f"""
:root {{
  --usf-green: {colors['primary']};
  --usf-dark-green: {colors['primary_dark']};
  --usf-light-bg: {colors['background']};
  --usf-secondary-bg: {colors['secondary']};
  --usf-text-color: {colors['text']};
}}
"""
    )
    if css_path.exists():
        chunks.append(css_path.read_text(encoding="utf-8"))
    st.markdown("<style>" + "\n".join(chunks) + "</style>", unsafe_allow_html=True)


def scroll_chat_to_bottom() -> None:
    """
    Improved auto-scroll logic that scrolls smoothly to the bottom of chat.
    Uses a more reliable approach with proper timing.
    """
    # Create a unique token based on current state to trigger scroll on changes
    messages_len = len(st.session_state.get("messages", []))
    show_email = int(bool(st.session_state.get("show_email_builder")))
    show_meeting = int(bool(st.session_state.get("show_meeting_builder")))
    show_picker = int(bool(st.session_state.get("show_tool_picker")))

    token = f"{messages_len}-{show_email}-{show_meeting}-{show_picker}"

    components.html(
        f"""
        <div style="display:none" data-scroll-token="{token}"></div>
        <script>
        (function() {{
            const scrollToBottom = () => {{
                try {{
                    // Try to access parent document (Streamlit iframe)
                    const doc = window.parent?.document || document;

                    // Target the main content container
                    const mainBlock = doc.querySelector('.main .block-container');
                    const scrollTarget = mainBlock || doc.documentElement || doc.body;

                    if (scrollTarget) {{
                        // Smooth scroll to bottom
                        scrollTarget.scrollTo({{
                            top: scrollTarget.scrollHeight,
                            behavior: 'smooth'
                        }});
                    }}
                }} catch (err) {{
                    // Fallback for security restrictions
                    console.debug('Scroll fallback:', err);
                    const fallback = document.documentElement || document.body;
                    fallback.scrollTop = fallback.scrollHeight;
                }}
            }};

            // Wait for DOM to be fully ready, then scroll
            if (document.readyState === 'complete') {{
                // Add small delay to ensure content is rendered
                setTimeout(scrollToBottom, 100);
            }} else {{
                window.addEventListener('load', () => {{
                    setTimeout(scrollToBottom, 100);
                }}, {{ once: true }});
            }}

            // Also try immediate scroll for fast updates
            setTimeout(scrollToBottom, 50);
        }})();
        </script>
        """,
        height=0,
    )
