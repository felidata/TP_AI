"""Stage 8: build report.html (self-contained; figures embedded) from results/ and figures/.

Every number in the report is read from a results file at build time; nothing is typed by hand.
Also writes provenance/numbers.json (one record per reported number).
"""
import base64, html, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from skbg import config  # noqa: E402

RUN = config.RUN_DIR
R = RUN / "results"; F = RUN / "figures"


def J(p):
    return json.loads((R / p).read_text())


def img(name, alt, caption, wide=False):
    p = F / name
    if not p.exists():
        return f'<p class="missing">[missing figure {html.escape(name)}]</p>'
    b = base64.b64encode(p.read_bytes()).decode()
    cls = "fig wide" if wide else "fig"
    return (f'<figure class="{cls}" id="fig-{p.stem}"><img src="data:image/png;base64,{b}" alt="{html.escape(alt)}" loading="lazy">'
            f'<figcaption><span class="figname">{html.escape(name)}</span> {caption}</figcaption></figure>')


def fmt(x, n=3):
    if x is None:
        return "—"
    if isinstance(x, (int,)) or (isinstance(x, float) and x.is_integer() and abs(x) > 100):
        return f"{int(x):,}".replace(",", " ")
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        m, e = f"{x:.{n - 1}e}".split("e")
        return f"{m}×10<sup>{int(e)}</sup>"
    return f"{x:.{n}f}"


def table(df, cols, heads, num_fmt=None, cls="data"):
    num_fmt = num_fmt or {}
    th = "".join(f"<th>{h}</th>" for h in heads)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            if c in num_fmt:
                v = num_fmt[c](v)
            elif isinstance(v, float):
                v = fmt(v)
            tds.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


CSS = """
:root{
  --paper:#f6f7f5; --panel:#ffffff; --ink:#15181c; --ink2:#4f5660; --rule:#d8dcd8; --accent:#2a78d6;
  --accent-ink:#1d5fae; --roi:#b87400; --roi-bg:#f6e7c8; --good:#1b7f4b; --bad:#b3372f; --code:#eef0ed;
  --serif:"IBM Plex Serif",Georgia,"Times New Roman",serif; --sans:"IBM Plex Sans Condensed","Arial Narrow",Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  color-scheme:dark; --paper:#121417; --panel:#1a1d21; --ink:#e7e9ec; --ink2:#a8afb8; --rule:#2e333a; --accent:#5a9be6;
  --accent-ink:#8cbaf0; --roi:#e0a33a; --roi-bg:#3a2e17; --good:#4cc38a; --bad:#ef7a70; --code:#20242a; } }
:root[data-theme="dark"]{ color-scheme:dark; --paper:#121417; --panel:#1a1d21; --ink:#e7e9ec; --ink2:#a8afb8; --rule:#2e333a;
  --accent:#5a9be6; --accent-ink:#8cbaf0; --roi:#e0a33a; --roi-bg:#3a2e17; --good:#4cc38a; --bad:#ef7a70; --code:#20242a; }
*{box-sizing:border-box}
body{background:var(--paper); color:var(--ink); font-family:var(--serif); font-size:17px; line-height:1.6; margin:0;
  padding-inline:clamp(16px,4vw,48px); padding-block:32px 80px}
.shell{max-width:1180px; margin:0 auto; display:grid; grid-template-columns:210px minmax(0,1fr); gap:48px}
nav.toc{position:sticky; top:calc(env(safe-area-inset-top,0px) + 24px); align-self:start; font-family:var(--sans); font-size:14px}
nav.toc ol{list-style:none; padding:0; margin:0; display:grid; gap:6px; counter-reset:s}
nav.toc a{color:var(--ink2); text-decoration:none} nav.toc a:hover{color:var(--accent-ink)}
nav.toc li{counter-increment:s} nav.toc li::before{content:counter(s) "  "; font-family:var(--mono); color:var(--ink2); font-size:12px}
main{min-width:0}
header.top{border-bottom:1px solid var(--rule); padding-bottom:20px; margin-bottom:28px}
.eyebrow{font-family:var(--sans); text-transform:uppercase; letter-spacing:.08em; font-size:13px; color:var(--ink2)}
h1{font-family:var(--sans); font-weight:600; font-size:clamp(30px,4vw,42px); line-height:1.12; margin:8px 0 12px; text-wrap:balance}
h2{font-family:var(--sans); font-weight:600; font-size:26px; margin:56px 0 12px; text-wrap:balance; scroll-margin-top:20px}
h3{font-family:var(--sans); font-weight:600; font-size:19px; margin:28px 0 8px}
p,li{max-width:68ch} .lede{font-size:19px; color:var(--ink)}
.meta{font-family:var(--mono); font-size:13px; color:var(--ink2); display:flex; flex-wrap:wrap; gap:6px 20px}
a{color:var(--accent-ink)}
code,.mono{font-family:var(--mono); font-size:.86em; background:var(--code); padding:1px 5px; border-radius:3px}
pre{font-family:var(--mono); font-size:13.5px; background:var(--code); padding:14px 16px; border-radius:4px; overflow-x:auto; max-width:100%}
.keynums{display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:1px; background:var(--rule);
  border:1px solid var(--rule); margin:22px 0 10px}
.keynums div{background:var(--panel); padding:14px 16px}
.keynums .k{font-family:var(--sans); font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink2)}
.keynums .v{font-family:var(--mono); font-size:24px; font-variant-numeric:tabular-nums; margin-top:4px}
.keynums .u{font-family:var(--mono); font-size:12.5px; color:var(--ink2)}
.tablewrap{overflow-x:auto; margin:14px 0 18px; max-width:100%}
table.data{border-collapse:collapse; font-family:var(--sans); font-size:14.5px; font-variant-numeric:tabular-nums; min-width:520px}
table.data th{text-align:left; font-weight:600; border-bottom:1.5px solid var(--ink2); padding:6px 12px 6px 0; color:var(--ink2)}
table.data td{border-bottom:1px solid var(--rule); padding:5px 12px 5px 0; vertical-align:top}
table.data td:not(:first-child){font-family:var(--mono); font-size:13.5px}
figure.fig{margin:22px 0 28px} figure.fig img{width:100%; max-width:100%; height:auto; display:block; background:#fff;
  border:1px solid var(--rule); border-radius:3px}
figcaption{font-family:var(--sans); font-size:14.5px; color:var(--ink2); margin-top:8px; max-width:80ch}
.figname{font-family:var(--mono); font-size:12px; color:var(--ink2); margin-right:6px}
.obs,.interp{border-left:3px solid var(--rule); padding:2px 0 2px 14px; margin:12px 0}
.obs{border-color:var(--accent)} .interp{border-color:var(--roi)}
.tag{font-family:var(--sans); font-size:12px; text-transform:uppercase; letter-spacing:.07em; color:var(--ink2); display:block}
.roi{background:var(--roi-bg); color:var(--ink); padding:0 4px; border-radius:2px}
.pass{color:var(--good); font-weight:600} .fail{color:var(--bad); font-weight:600}
.missing{color:var(--bad); font-family:var(--mono)}
footer{margin-top:64px; border-top:1px solid var(--rule); padding-top:16px; font-family:var(--sans); font-size:14px; color:var(--ink2)}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
@media (max-width:900px){ .shell{grid-template-columns:1fr; gap:0} nav.toc{position:static; margin-bottom:24px} }
"""

SECTIONS = [("summary", "Summary"), ("motivation", "Motivation"), ("data", "Data"), ("diagnostics", "Initial diagnostics"),
            ("research", "Research process"), ("method", "Chosen method"), ("validation", "Validation design"),
            ("results", "Results"), ("tradeoff", "Exposure–background trade-off"), ("checks", "Checks"),
            ("limits", "Limitations and failures"), ("provenance", "Provenance"), ("repro", "Reproduction")]


def build(body_sections, meta):
    toc = "".join(f'<li><a href="#{k}">{v}</a></li>' for k, v in SECTIONS)
    fonts = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&'
             'family=IBM+Plex+Sans+Condensed:wght@400;600&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Skipper-CCD Low-Energy Background Masks</title>{fonts}<style>{CSS}</style></head><body>
<div class="shell"><nav class="toc" aria-label="Contents"><ol>{toc}</ol></nav><main>
{body_sections}
<footer>Run <span class="mono">{meta['run_id']}</span> · commit <span class="mono">{meta['commit']}</span> · generated by <span class="mono">scripts/08_report.py</span> from <span class="mono">results/</span> and <span class="mono">figures/</span>.</footer>
</main></div></body></html>"""


if __name__ == "__main__":
    import importlib.util
    spec = importlib.util.spec_from_file_location("report_body", RUN / "scripts" / "report_body.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    body, meta, numbers = mod.body(J, img, fmt, table, pd, R)
    (RUN / "report.html").write_text(build(body, meta))
    (RUN / "provenance").mkdir(exist_ok=True)
    (RUN / "provenance" / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print("wrote report.html and provenance/numbers.json")
