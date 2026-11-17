"""Plot the measured C0 footprint and controls on the frozen 10° square."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def render_comparison(data, output):
    """Show paired precision and the frozen physical-run sensitivity intervals."""
    bands = ("ordinary", "transition", "repeat_band")
    labels = ("Ordinary", "Transition", "Repeat band")
    methods = (("NSC", "Official NSC", "#687581"),
               ("Xiao", "Xiao 2025 ground", "#ce843d"),
               ("C0", "SDSS + NSC C0", "#24689b"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for column, magnitude in enumerate(("21_to_22", "22_to_23")):
        precision, gain = axes[:, column]
        for index, band in enumerate(bands):
            sample = data["cohorts"].get(band, {}).get("bins", {}).get(magnitude, {}).get("common", {})
            count = sample.get("n", 0)
            precision.text(index, 1.02, f"n={count}", transform=precision.get_xaxis_transform(),
                           ha="center", fontsize=10)
            for method_index, (name, label, color) in enumerate(methods):
                value = sample.get("methods", {}).get(name, {}).get("radial_p68")
                if value is not None:
                    precision.scatter(index + (method_index-1)*.15, value, color=color, s=45)
            for method_index, (name, label, color) in enumerate(methods[:2]):
                estimate = sample.get("paired_radial_gain", {}).get(name, {})
                value = estimate.get("relative_gain")
                if value is None:
                    continue
                x = index + (method_index-.5)*.15
                gain.scatter(x, 100*value, color=color, s=45)
                interval = estimate.get("visit_weighted", {}).get("sdss_run", {}).get("relative_gain95")
                if interval is not None:
                    gain.vlines(x, 100*interval[0], 100*interval[1], color=color, linewidth=2)
        precision.set(title=magnitude.replace("_to_", " ≤ r < "), ylabel="Radial P68 [mas/yr]", ylim=(0, None))
        gain.axhline(0, color="#444444", linewidth=1)
        gain.set(ylabel="C0 relative P68 improvement [%]")
        for axis in (precision, gain):
            axis.set(xticks=range(len(bands)), xticklabels=labels, xlim=(-.5, 2.5))
            axis.grid(axis="y", alpha=.2)
        if column == 0:
            precision.legend(handles=[Line2D([], [], color=color, marker="o", linestyle="none", label=label)
                                      for _, label, color in methods], fontsize=9)
            gain.legend(handles=[Line2D([], [], color=color, marker="o", linestyle="none",
                                       label=f"C0 vs {label}") for _, label, color in methods[:2]], fontsize=9)
    fig.suptitle("Strict common spectroscopic QSOs · C0 uses the shared continuous reference graph")
    fig.supxlabel("Intervals show 95% SDSS-run weight sensitivity · positive values favor C0", fontsize=10)
    fig.savefig(output/"v12_continuous_paired_precision_summary.png", dpi=170)
    plt.close(fig)


def render(data, output, detailed=False):
    cores=data["cores"]
    boxes=data["covered_cores"]
    p68={}
    zero={}
    support={}
    count={}
    repeat={}
    errors={}
    north={}
    east_coverage={}
    north_coverage={}
    visits={}
    nsc_span={}
    sdss_span={}
    for k,item in cores.items():
        q=item["qso"]["by_cohort"].get("c0",{}).get("strict__all__22_to_23",{})
        c=q.get("methods",{}).get("C0",{})
        p68[k]=(c.get("radial_abs68"),q.get("n",0))
        median=c.get("median",[None,None])
        zero[k]=(median[0],q.get("n",0))
        north[k]=(median[1],q.get("n",0))
        coverage=c.get("fraction_within_1sigma",[None,None])
        east_coverage[k]=(coverage[0],q.get("n",0))
        north_coverage[k]=(coverage[1],q.get("n",0))
        ns=item["observations"]["NSC"]
        sd=item["observations"]["SDSS"]
        nsc_span[k]=(ns["span_years"],None)
        sdss_span[k]=(sd["span_years"],None)
        d=ns["n_rows"]+ns["unconstrained"]+sd["n_rows"]+sd["unconstrained"]
        support[k]=((ns["n_rows"]+sd["n_rows"])/d*100 if d else None,None)
        count[k]=(item["owned_w0"]/boxes[k]["area_sqdeg"],None)
        repeat[k]=(item["leverage"]["repeat_sdss"]["n"]/item["owned_w0"]*100 if item["owned_w0"] else None,None)
        err=item["formal_error_masyr"]
        errors[k]=(max(err[1]) if err is not None else None,None)
        stats=item["leverage"]["all"]["independent_groups_flags_downweight_leverage_p16_50_84"]
        visits[k]=(stats[1][0] if stats is not None else None,None)

    def upper(mapped):
        return max((v for v,_ in mapped.values() if v is not None),default=1) or 1

    zero_limit=max((abs(v) for axis in (zero,north) for v,_ in axis.values() if v is not None),
                   default=1) or 1
    zero_norm=TwoSlopeNorm(0,vmin=-zero_limit,vmax=zero_limit)
    panels=[
        ("W0 sources / deg²",count,"viridis",Normalize(0,upper(count))),
        ("Supported r detections [%]",support,"viridis",Normalize(0,100)),
        ("W0 sources with repeated SDSS [%]",repeat,"viridis",Normalize(0,100)),
        ("QSO radial P68 [mas/yr]",p68,"viridis",Normalize(0,upper(p68))),
        ("QSO east zero [mas/yr]",zero,"RdBu_r",zero_norm),
        ("Median formal axis error [mas/yr]",errors,"viridis",Normalize(0,upper(errors))),
    ]
    if detailed:
        coverage_norm=TwoSlopeNorm(.682689,vmin=0,vmax=1)
        panels.extend([
            ("QSO north zero [mas/yr]",north,"RdBu_r",zero_norm),
            ("Median independent groups per W0",visits,"viridis",Normalize(0,upper(visits))),
            ("QSO inside east 1σ [fraction]",east_coverage,"RdBu",coverage_norm),
            ("QSO inside north 1σ [fraction]",north_coverage,"RdBu",coverage_norm),
            ("NSC supported r span [yr]",nsc_span,"viridis",Normalize(0,upper(nsc_span))),
            ("SDSS supported r span [yr]",sdss_span,"viridis",Normalize(0,upper(sdss_span))),
        ])
    columns=3
    rows=len(panels)//columns
    fig, ax=plt.subplots(rows,columns,figsize=(12,3.8*rows),layout="constrained")
    for a, (title, mapped, cmap, norm) in zip(ax.ravel(),panels):
        colors=plt.get_cmap(cmap)
        a.add_patch(Rectangle((345,-5),10,10,facecolor="#eeeeee",edgecolor="none"))
        for k,(v,n) in mapped.items():
            box=boxes[k]
            lo, hi=box["ra_bounds_deg"]
            south,north=box["dec_bounds_deg"]
            a.add_patch(Rectangle((lo,south),hi-lo,north-south,
                                  facecolor=colors(norm(v)) if v is not None else "#d4d4d4",
                                  edgecolor="white",linewidth=1))
            if n is not None:
                a.text((lo+hi)/2,(south+north)/2,str(n),ha="center",va="center",fontsize=8,
                       bbox=dict(facecolor="white",alpha=.85,edgecolor="none",pad=1))
        a.set(xlim=(355,345),ylim=(-5,5),xticks=(345,347,349,351,353,355),
              yticks=(-5,-3,-1,1,3,5),aspect="equal",xlabel="RA [deg]",ylabel="Dec [deg]",title=title)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=colors),ax=a,shrink=.65)
    fig.suptitle("Continuous C field · labels: QSO counts at 22 ≤ r < 23 · gray: missing metric",fontsize=12)
    filename="v12_continuous_science_summary.png" if detailed else "v12_continuous_coverage_summary.png"
    fig.savefig(output/filename,dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary",type=Path)
    p.add_argument("figures",type=Path)
    p.add_argument("--additional",nargs="*",type=Path,default=[])
    p.add_argument("--detailed",action="store_true",help="Include north zero, visit support and axis error coverage")
    p.add_argument("--comparison",action="store_true",help="Also plot strict common QSO precision and paired gains")
    a=p.parse_args()
    a.figures.mkdir(parents=True,exist_ok=True)
    data=json.loads(a.summary.read_text())
    for path in a.additional:
        extra=json.loads(path.read_text())
        for name in ("cores","covered_cores"):
            if data[name].keys() & extra[name].keys():
                p.error("Each verified core must appear once in the map")
            data[name].update(extra[name])
    render(data,a.figures,a.detailed)
    if a.comparison:
        render_comparison(data,a.figures)


if __name__=="__main__":
    main()
